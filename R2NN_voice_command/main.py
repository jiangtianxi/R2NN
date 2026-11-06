import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import metarnn
from scipy import io
from torch.utils.data.dataset import Dataset
from torch.autograd import Variable
import datetime
import copy
import inspect
from pathlib import Path
from metarnn import generateCoupMat
from metarnn import construct
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset
import gc 

# Use double precision throughout the simulation and optimization. Long
# second-order recurrences are noticeably more sensitive to round-off than
# ordinary feed-forward networks.
torch.set_default_dtype(torch.float64)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print(f"Compute device: {device} (global dtype: float64)")

# Prefer deterministic CuDNN kernels when CUDA is available.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Source, fixed-node and probe IDs are derived from these dimensions below.
GRID_ROWS = 5
GRID_COLS = 5

# Select one physical dynamics mode:
#   linear      : no local nonlinear restoring force;
#   unilateral : trainable one-sided (diode-like) local springs;
#   cubic       : fixed cubic local restoring force.
# "nonlinear" and "diode" are accepted as aliases for "unilateral".
model_mode = "linear"
model_mode_aliases = {
    "nonlinear": "unilateral",
    "diode": "unilateral",
}
model_mode = model_mode_aliases.get(model_mode, model_mode)
if model_mode not in {"linear", "unilateral", "cubic"}:
    raise ValueError(
        "model_mode must be 'linear', 'unilateral' or 'cubic'")
use_unilateral = model_mode == "unilateral"
use_cubic = model_mode == "cubic"
use_nonlinear = use_unilateral or use_cubic
print(f"Physical model mode: {model_mode}")

# Step 1: load the input sequences and labels.
class SignalClassificationDataset(Dataset):
    """Waveforms ``x`` with shape (n_samples, N) and integer labels ``y``."""
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


time_now = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')
train_log_filename = (
    f"train_log_{model_mode}_double_{time_now}.txt")
train_log_filepath = "./log/"+ train_log_filename
train_log_txt_formatter = (
    "{time_str} [Fold] {fold} [Epoch] {epoch:03d} [proportion] {prop} "
    "[Lr(kn,kc)*1e5] {lr_linear} [Lr(kd)*1e6] {lr_nonlinear} "
    "[Loss] {loss_str} [GradNorm] {grad_norm} "
    "[dt2LambdaMax] {dt2_lambda_max} [ProjectionScale] {projection_scale} "
    "[GradNorm(kn,kc)] {linear_grad_norm} "
    "[GradNorm(kd)] {nonlinear_grad_norm} "
    "[CELoss] {ce_loss} [MarginLoss] {margin_loss}\n"
)

# Each MAT file stores one spoken class as Sample[time, sample]. Labels 0/1/2
# correspond to hou/qian/ting, to sample1/sample2/sample3, and to the three
# output probes in the same order. Only the training split is used for CV and
# model selection; the independent test split remains locked until final-model
# evaluation.
NUM_CLASSES = 3
CLASS_NAMES = ("hou", "qian", "ting")
EXPECTED_SEQUENCE_LENGTH = 10000
EXPECTED_SAMPLES_PER_CLASS = {
    "train": 80,
    "test": 20,
}
dataset_root = Path(__file__).resolve().parent / "data"

if not all(
        (dataset_root / "train" / f"sample{class_id}.mat").is_file()
        for class_id in range(1, NUM_CLASSES + 1)):
    raise FileNotFoundError(
        "Could not find sample1.mat, sample2.mat and sample3.mat under "
        f"{dataset_root / 'train'}")


def load_classification_split(
        split_directory, split_name, expected_samples_per_class):
    """Load one balanced, fixed-length three-class waveform split."""
    signal_parts = []
    label_parts = []
    class_counts = []

    for class_label in range(NUM_CLASSES):
        data_file = split_directory / f"sample{class_label + 1}.mat"
        if not data_file.is_file():
            raise FileNotFoundError(f"Missing {split_name} file: {data_file}")
        content = io.loadmat(str(data_file))
        if "Sample" not in content:
            raise KeyError(f"{data_file} must contain 'Sample'")

        sample_matrix = np.asarray(content["Sample"], dtype=np.float64)
        if sample_matrix.ndim != 2:
            raise ValueError(
                f"{data_file}: Sample must be a 2-D [time, sample] matrix")
        class_signals = sample_matrix.T
        if class_signals.shape[1] != EXPECTED_SEQUENCE_LENGTH:
            raise ValueError(
                f"{data_file}: expected sequence length "
                f"{EXPECTED_SEQUENCE_LENGTH}, found "
                f"N={class_signals.shape[1]}")
        if class_signals.shape[0] != expected_samples_per_class:
            raise ValueError(
                f"{data_file}: expected {expected_samples_per_class} "
                f"samples, found {class_signals.shape[0]}")
        if not np.isfinite(class_signals).all():
            raise ValueError(
                f"{data_file}: Sample contains NaN or infinite values")

        signal_parts.append(class_signals)
        label_parts.append(np.full(
            class_signals.shape[0], class_label, dtype=np.int64))
        class_counts.append(class_signals.shape[0])

    split_x = torch.from_numpy(
        np.concatenate(signal_parts, axis=0)).double()
    split_y = torch.from_numpy(
        np.concatenate(label_parts)).long()
    if len(set(class_counts)) != 1:
        raise ValueError(
            f"{split_name} split must be class-balanced; got {class_counts}")

    print(
        f"Loaded {split_name} split from {split_directory}: "
        f"samples={len(split_y)}, N={split_x.shape[1]}, "
        f"per-class={class_counts}")
    return SignalClassificationDataset(split_x, split_y), split_y


fullDataset, y = load_classification_split(
    dataset_root / "train",
    "training",
    EXPECTED_SAMPLES_PER_CLASS["train"])


def require_disjoint_waveforms(
        first_dataset, second_dataset, first_name, second_name):
    """Reject exact waveform reuse between two independently made splits."""
    first_rows = {
        np.ascontiguousarray(row.numpy()).tobytes()
        for row in first_dataset.x
    }
    overlap_count = sum(
        np.ascontiguousarray(row.numpy()).tobytes() in first_rows
        for row in second_dataset.x)
    if overlap_count:
        raise ValueError(
            f"{first_name} and {second_name} contain "
            f"{overlap_count} exactly duplicated waveform(s)")
    print(f"Verified: {first_name} and {second_name} are disjoint")


# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
batch_size = 32
max_epoch = 60
# Lower bound for the unilateral stiffness k_d.
min_stiffness = 1e-8
# Admissible kn/kc values are {0} union [lower bound, upper bound].
linear_stiffness_zero_threshold = 0
linear_stiffness_maximum = 5e-3
energy_epsilon = 1e-12
# Clip the linear and nonlinear parameter groups independently so a large
# kn/kc gradient does not reduce the kd update in the same batch.
max_linear_grad_norm = 5e3
max_nonlinear_grad_norm = 300.0
# Stability bound on dt^2 * lambda_max(M^-1 K); the central-difference
# scheme requires a value below 4.
max_dt2_eigenvalue = 4.0
# Early stopping: stop after this many epochs without a validation-CE
# improvement larger than early_stopping_min_delta.
early_stopping_patience = 15
early_stopping_min_delta = 1e-4
model_selection_accuracy_tolerance = 1e-12
# Unilateral mode: temperature of the sigmoid surrogate gradient used for
# the spring switch (the forward pass always uses the hard switch).
switch_temperature = 1.0
use_surrogate_gradient = True
# Cubic mode: displacement clamp that prevents numerical overflow.
state_limit = 1.0e6
# Cubic mode: fixed coefficient k3 of the local restoring force k3*delta^3.
fixed_cubic_coefficient = 10.0e-10
# Use one identical classification objective for all physical modes so the
# linear, unilateral and cubic comparisons differ only in their dynamics.
# Margin loss: penalise samples whose true-class energy is less than
# energy_margin_ratio times the largest other-class energy, weighted by
# energy_margin_weight.
energy_margin_ratio = 1.2
energy_margin_weight = 0.00
print(
    "Training configuration: margin_ratio={:.1f}, "
    "ordinary sample-mean loss for every physical mode".format(
        energy_margin_ratio))
print(
    "Linear stiffness constraint: kn/kc in {{0}} union "
    "[{:.6e}, {:.6e}]".format(
        linear_stiffness_zero_threshold, linear_stiffness_maximum))
if use_cubic:
    print(
        "Fixed cubic coefficient: k3={:.6e}, state_limit={:.1f}"
        .format(fixed_cubic_coefficient, state_limit))


def require_finite(name, value, fold, epoch, batch_idx):
    """Raise FloatingPointError if ``value`` contains NaN or Inf."""
    if not torch.isfinite(value).all():
        raise FloatingPointError(
            f"{name} became non-finite at fold={fold}, epoch={epoch}, "
            f"batch={batch_idx + 1}")


def gradient_l2_norm(parameters):
    """Return the L2 norm of a selected parameter group's gradients."""
    squared_norms = [
        parameter.grad.detach().square().sum()
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not squared_norms:
        return torch.zeros((), device=device)
    return torch.sqrt(torch.stack(squared_norms).sum())


@torch.no_grad()
def nonlinear_coefficient_summary(model):
    """Return compact diagnostics for the selected nonlinear coefficient."""
    if use_cubic:
        coefficient = model.cell.coup.k_cubic.detach()
        coefficient_name = "k_cubic(fixed)"
    else:
        coefficient = model.cell.coup.k_d.detach()
        coefficient_name = "k_d"
    return {
        "name": coefficient_name,
        "minimum": coefficient.min().item(),
        "mean": coefficient.mean().item(),
        "maximum": coefficient.max().item(),
    }


def signed_log_energy_margin_per_sample(logits, target):
    """Return log(E_true / E_other) for every sample."""
    target = target.long()
    true_logits = logits.gather(1, target.unsqueeze(1)).squeeze(1)
    target_mask = F.one_hot(
        target, num_classes=logits.shape[1]).to(dtype=torch.bool)
    other_logits = logits.masked_fill(
        target_mask, float("-inf")).max(dim=1).values
    return true_logits - other_logits


def classification_loss_per_sample(logits, target):
    """Return per-sample CE and log-energy-ratio margin losses."""
    target = target.long()
    ce_loss = F.cross_entropy(logits, target, reduction="none")
    signed_margin = signed_log_energy_margin_per_sample(logits, target)

    log_margin = logits.new_tensor(np.log(energy_margin_ratio))
    margin_loss = F.relu(log_margin - signed_margin)
    return ce_loss, margin_loss


def training_classification_loss(logits, target):
    """Return the sample-mean loss used for training and evaluation."""
    ce_per_sample, margin_per_sample = classification_loss_per_sample(
        logits, target)
    ce_loss = ce_per_sample.mean()
    margin_loss = margin_per_sample.mean()
    total_loss = ce_loss + energy_margin_weight * margin_loss
    return total_loss, ce_loss, margin_loss


@torch.no_grad()
def project_stiffness_with_compatibility(model):
    """
    Bound kn/kc to {0} union [lower, upper] and enforce stability.

    Coupling.project_stiffness_ performs the projection when it accepts
    dt and max_dt2_eigenvalue. Otherwise the equivalent projection below
    is applied.
    """
    coupling = model.cell.coup
    include_unilateral_stiffness = (
        getattr(model.cell, "nonlinearity_mode", "unilateral")
        == "unilateral")
    linear_stiffnesses = (coupling.k_n, coupling.k_c)
    for value in linear_stiffnesses:
        value.clamp_(max=linear_stiffness_maximum)
        value.masked_fill_(
            value < linear_stiffness_zero_threshold, 0.0)

    projection_method = getattr(coupling, "project_stiffness_", None)
    if projection_method is not None:
        method_parameters = inspect.signature(projection_method).parameters
        if {"dt", "max_dt2_eigenvalue"}.issubset(method_parameters):
            projection_arguments = dict(
                min_value=min_stiffness,
                dt=model.cell.dt,
                max_dt2_eigenvalue=max_dt2_eigenvalue)
            if "linear_zero_threshold" in method_parameters:
                projection_arguments["linear_zero_threshold"] = (
                    linear_stiffness_zero_threshold)
            if "include_nonlinear" in method_parameters:
                projection_arguments["include_nonlinear"] = (
                    include_unilateral_stiffness)
            if "fixed_indices" in method_parameters:
                projection_arguments["fixed_indices"] = model.fxp
            stats = projection_method(**projection_arguments)
            # Reapply both bounds after stability scaling.
            for value in linear_stiffnesses:
                value.clamp_(max=linear_stiffness_maximum)
                value.masked_fill_(
                    value < linear_stiffness_zero_threshold, 0.0)
            return stats

    nonlinear_stiffnesses = (
        (coupling.k_d,) if include_unilateral_stiffness else ())
    stiffnesses = linear_stiffnesses + nonlinear_stiffnesses
    if not all(torch.isfinite(value).all() for value in stiffnesses):
        raise FloatingPointError("Non-finite trainable stiffness detected")
    for value in nonlinear_stiffnesses:
        value.clamp_(min=min_stiffness)

    kmat, mmat, _ = construct.construct_MKC(
        coupling.k_n, coupling.k_c,
        coupling.m_n, coupling.m_c,
        coupling.c_n, coupling.c_c,
        coupling.coupMat, coupling.c_g)

    mass_diagonal = torch.diagonal(mmat)
    if (not torch.isfinite(mass_diagonal).all()
            or torch.any(mass_diagonal <= 0)):
        raise FloatingPointError(
            "Masses must be finite and positive for stability projection")

    kdelta = torch.zeros_like(kmat)
    if include_unilateral_stiffness:
        n_local = coupling.k_d.numel()
        idx_main = torch.arange(n_local, device=kmat.device)
        idx_local = idx_main + n_local
        kd = coupling.k_d.reshape(-1)
        kdelta[idx_main, idx_main] = kd
        kdelta[idx_local, idx_local] = kd
        kdelta[idx_main, idx_local] = -kd
        kdelta[idx_local, idx_main] = -kd

    inverse_sqrt_mass = torch.rsqrt(mass_diagonal)
    normalized_stiffness = (
        inverse_sqrt_mass[:, None]
        * (kmat + kdelta)
        * inverse_sqrt_mass[None, :])
    normalized_stiffness = (
        normalized_stiffness + normalized_stiffness.transpose(0, 1)) / 2
    fixed_indices = torch.as_tensor(
        model.fxp, device=kmat.device, dtype=torch.long).reshape(-1)
    free_mask = torch.ones(
        normalized_stiffness.shape[0],
        device=kmat.device, dtype=torch.bool)
    free_mask[fixed_indices] = False
    free_indices = torch.arange(
        normalized_stiffness.shape[0],
        device=kmat.device)[free_mask]
    normalized_stiffness = normalized_stiffness.index_select(
        0, free_indices).index_select(1, free_indices)
    lambda_max = torch.linalg.eigvalsh(normalized_stiffness).max()
    dt = torch.as_tensor(
        model.cell.dt, device=kmat.device, dtype=kmat.dtype)
    dt2_lambda_max = dt.square() * lambda_max
    if not torch.isfinite(dt2_lambda_max):
        raise FloatingPointError(
            "Non-finite eigenvalue in stiffness stability check")

    projection_scale = torch.ones_like(dt2_lambda_max)
    if dt2_lambda_max > max_dt2_eigenvalue:
        projection_scale = (
            torch.as_tensor(
                max_dt2_eigenvalue,
                device=kmat.device,
                dtype=kmat.dtype)
            / dt2_lambda_max)
        for value in stiffnesses:
            value.mul_(projection_scale)
        for value in linear_stiffnesses:
            value.clamp_(max=linear_stiffness_maximum)
            value.masked_fill_(
                value < linear_stiffness_zero_threshold, 0.0)
        for value in nonlinear_stiffnesses:
            value.clamp_(min=min_stiffness)
        dt2_lambda_max = dt2_lambda_max * projection_scale

    return {
        "dt2_lambda_max": dt2_lambda_max.item(),
        "projection_scale": projection_scale.item(),
    }


def train(model, train_loader, optimizer, epoch, max_epoch, fold, lr_list, loss_iter):
    """Run one training epoch and return the sample-weighted mean loss.

    For each batch: simulate the lattice, sum the squared probe responses over
    time (class energies), compute the loss on log energies, back-propagate,
    clip the gradients of each parameter group, take an optimizer step and
    project the stiffnesses back to the admissible, stable set. Every batch
    is written to the training log.
    """
    model.train()
    total_loss = 0.0
    total_samples = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        X = data.unsqueeze(2)
        temp_v = model(X)
        # Class energy = time-integrated squared probe displacement.
        energy = temp_v.sum(dim=1)
        if energy.ndim != 2 or energy.shape[1] != NUM_CLASSES:
            raise ValueError(
                f"Expected {NUM_CLASSES} probe-energy outputs, got "
                f"shape {tuple(energy.shape)}")
        require_finite("energy", energy, fold, epoch, batch_idx)
        logits = metarnn.utils.energy_to_logits(energy, eps=energy_epsilon)
        require_finite("logits", logits, fold, epoch, batch_idx)
        loss, ce_loss, margin_loss = training_classification_loss(
            logits, target)
        require_finite("loss", loss, fold, epoch, batch_idx)
        require_finite(
            "cross-entropy loss", ce_loss, fold, epoch, batch_idx)
        require_finite(
            "energy margin loss", margin_loss, fold, epoch, batch_idx)
        loss.backward()
        linear_grad_norm = gradient_l2_norm([
            model.cell.coup.k_n,
            model.cell.coup.k_c,
        ])
        nonlinear_grad_norm = gradient_l2_norm([
            model.cell.coup.k_d,
        ])
        require_finite(
            "kn/kc gradient norm", linear_grad_norm,
            fold, epoch, batch_idx)
        require_finite(
            "kd gradient norm", nonlinear_grad_norm,
            fold, epoch, batch_idx)

        # Preserve the pre-clipping total norm for diagnostics. The two
        # optimizer groups are clipped separately below.
        grad_norm = torch.sqrt(
            linear_grad_norm.square() + nonlinear_grad_norm.square())
        require_finite("gradient norm", grad_norm, fold, epoch, batch_idx)

        torch.nn.utils.clip_grad_norm_(
            [
                model.cell.coup.k_n,
                model.cell.coup.k_c,
            ],
            max_norm=max_linear_grad_norm,
        )
        if use_unilateral:
            torch.nn.utils.clip_grad_norm_(
                [model.cell.coup.k_d],
                max_norm=max_nonlinear_grad_norm,
            )

        optimizer.step()
        # Keep the updated stiffnesses physical and the integration stable.
        stability_stats = project_stiffness_with_compatibility(model)

        # Record the optimizer learning rates used for this batch.
        linear_lr = optimizer.param_groups[0]['lr']
        nonlinear_lr = (
            optimizer.param_groups[1]['lr']
            if len(optimizer.param_groups) > 1
            else 0.0)
        lr_list.append((linear_lr, nonlinear_lr))

        print("Fold: {} -- Epoch: {} -- [{}/{} ({}%)] "
              "-- Lr(kn,kc): {:.3e} -- Lr(kd): {:.3e} "
              "-- Loss(total/CE/margin): {:.7f}/{:.7f}/{:.7f} "
              "-- GradNorm(total/linear/kd): "
              "{:.3e}/{:.3e}/{:.3e} "
              "-- dt2LambdaMax: {:.4f}".format(
            fold, epoch, (batch_idx + 1), len(train_loader),
            100. * (batch_idx + 1) / len(train_loader),
            linear_lr, nonlinear_lr,
            loss.item(), ce_loss.item(), margin_loss.item(),
            grad_norm.item(),
            linear_grad_norm.item(), nonlinear_grad_norm.item(),
            stability_stats["dt2_lambda_max"]))
        
        to_write = train_log_txt_formatter.format(
            time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S'),
            fold = fold,
            epoch = epoch,
            prop = "{}/{}".format((batch_idx + 1), len(train_loader)),
            lr_linear = "{:.4f}".format(linear_lr * 1e5),
            lr_nonlinear = "{:.4f}".format(nonlinear_lr * 1e6),
            loss_str = "{:.7f}".format(loss),
            grad_norm = "{:.6e}".format(grad_norm.item()),
            dt2_lambda_max = "{:.6f}".format(
                stability_stats["dt2_lambda_max"]),
            projection_scale = "{:.6f}".format(
                stability_stats["projection_scale"]),
            linear_grad_norm = "{:.6e}".format(
                linear_grad_norm.item()),
            nonlinear_grad_norm = "{:.6e}".format(
                nonlinear_grad_norm.item()),
            ce_loss = "{:.7f}".format(ce_loss.item()),
            margin_loss = "{:.7f}".format(margin_loss.item()))
        
        with open(train_log_filepath,"a") as f:
            f.write(to_write)
        loss_iter.append(loss.item())
        total_loss += loss.item() * data.shape[0]
        total_samples += data.shape[0]
        
    return total_loss / total_samples

def evaluate(model, data_loader, epoch, fold, split_name):
    """
    Evaluate CE, margin, total loss and accuracy with one fixed model.

    The same loss definitions are used for train, validation and test
    evaluation. No parameter update is performed here.
    """
    model.eval()
    summed_ce_loss = 0.0
    summed_margin_loss = 0.0
    correct = 0
    total_samples = 0
    class_correct = np.zeros(NUM_CLASSES, dtype=np.int64)
    class_samples = np.zeros(NUM_CLASSES, dtype=np.int64)
    class_signed_margin_sum = np.zeros(NUM_CLASSES, dtype=np.float64)
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            data = data.to(device)
            target = target.to(device)

            X = data.unsqueeze(2)
            probe_energy_over_time = model(X)
            energy = probe_energy_over_time.sum(dim=1)
            if energy.ndim != 2 or energy.shape[1] != NUM_CLASSES:
                raise ValueError(
                    f"Expected {NUM_CLASSES} probe-energy outputs, got "
                    f"shape {tuple(energy.shape)}")
            require_finite(
                f"{split_name} energy", energy, fold, epoch, batch_idx)
            logits = metarnn.utils.energy_to_logits(
                energy, eps=energy_epsilon)
            require_finite(
                f"{split_name} logits", logits, fold, epoch, batch_idx)
            signed_margin = signed_log_energy_margin_per_sample(
                logits, target)
            require_finite(
                f"{split_name} signed log-energy margin",
                signed_margin, fold, epoch, batch_idx)
            probability = metarnn.utils.normalize_power(
                energy, eps=energy_epsilon)
            batch_total_loss, batch_ce_loss, batch_margin_loss = (
                training_classification_loss(logits, target))
            require_finite(
                f"{split_name} total loss",
                batch_total_loss, fold, epoch, batch_idx)
            require_finite(
                f"{split_name} cross-entropy loss",
                batch_ce_loss, fold, epoch, batch_idx)
            require_finite(
                f"{split_name} energy margin loss",
                batch_margin_loss, fold, epoch, batch_idx)

            current_batch_size = data.shape[0]
            summed_ce_loss += (
                batch_ce_loss.item() * current_batch_size)
            summed_margin_loss += (
                batch_margin_loss.item() * current_batch_size)
            pred = probability.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total_samples += current_batch_size
            encoded_pairs = target * NUM_CLASSES + pred
            confusion += (
                torch.bincount(
                    encoded_pairs, minlength=NUM_CLASSES**2)
                .reshape(NUM_CLASSES, NUM_CLASSES)
                .cpu()
                .numpy())
            for class_label in range(NUM_CLASSES):
                class_mask = target == class_label
                class_count = class_mask.sum().item()
                class_samples[class_label] += class_count
                if class_count:
                    class_signed_margin_sum[class_label] += (
                        signed_margin[class_mask].sum().item())
                    class_correct[class_label] += (
                        pred[class_mask]
                        .eq(target[class_mask])
                        .sum()
                        .item())

    average_ce_loss = summed_ce_loss / total_samples
    average_margin_loss = summed_margin_loss / total_samples
    average_total_loss = (
        average_ce_loss
        + energy_margin_weight * average_margin_loss)
    accuracy = 100.0 * correct / total_samples

    # "Average loss" is the CE loss. Checkpoints are selected by validation
    # accuracy, with validation CE as the tie-breaker.
    summary = (
        f"{split_name} set (Fold {fold}, Epoch {epoch}): "
        f"Average loss: {average_ce_loss:.4f}, "
        f"Accuracy: {correct}/{total_samples} ({accuracy:.2f}%)\n")
    component_summary = (
        f"{split_name} loss components "
        f"(Fold {fold}, Epoch {epoch}): "
        f"CE={average_ce_loss:.6f}, "
        f"Margin={average_margin_loss:.6f}, "
        f"Total={average_total_loss:.6f}\n")
    class_accuracy = {
        CLASS_NAMES[class_label]: (
            100.0 * class_correct[class_label] / class_samples[class_label]
            if class_samples[class_label]
            else float("nan"))
        for class_label in range(NUM_CLASSES)
    }
    class_signed_margin = {
        CLASS_NAMES[class_label]: (
            class_signed_margin_sum[class_label] / class_samples[class_label]
            if class_samples[class_label]
            else float("nan"))
        for class_label in range(NUM_CLASSES)
    }
    minimum_class_accuracy = min(class_accuracy.values())
    class_summary = (
        f"{split_name} per-class accuracy "
        f"(Fold {fold}, Epoch {epoch}): "
        + ", ".join(
            f"{class_name}={class_value:.2f}%"
            for class_name, class_value in class_accuracy.items())
        + "\n")
    class_margin_summary = (
        f"{split_name} per-class signed log-energy margin "
        f"(Fold {fold}, Epoch {epoch}): "
        + ", ".join(
            f"{class_name}={margin:.6f}"
            for class_name, margin in class_signed_margin.items())
        + "\n")
    confusion_summary = (
        f"{split_name} confusion matrix (rows=true, columns=predicted; "
        f"order={list(CLASS_NAMES)}): "
        + "; ".join(
            f"{CLASS_NAMES[class_label]}={confusion[class_label].tolist()}"
            for class_label in range(NUM_CLASSES))
        + "\n")
    print(
        summary + component_summary + class_summary
        + class_margin_summary + confusion_summary,
        end="")
    with open(train_log_filepath, "a") as f:
        f.write(
            summary + component_summary + class_summary
            + class_margin_summary + confusion_summary)

    return {
        "ce_loss": average_ce_loss,
        "margin_loss": average_margin_loss,
        "total_loss": average_total_loss,
        "accuracy": accuracy,
        "correct": correct,
        "samples": total_samples,
        "class_accuracy": class_accuracy,
        "minimum_class_accuracy": minimum_class_accuracy,
        "class_signed_margin": class_signed_margin,
        "confusion_matrix": confusion,
    }


def create_model_optimizer_scheduler():
    """Create one fresh model with the common optimizer configuration."""
    beta = 1
    dt = 1
    row, col = GRID_ROWS, GRID_COLS
    # Two degrees of freedom per unit cell: main mass and local resonator.
    dim = 2 * row * col
    domain_shape = (dim, 1)

    # Construct every physical parameter in float64 on the target device.
    m_c = (
        3.0e-2
        * torch.ones(1, row * col, dtype=torch.float64)
    ).to(device)
    m_n = (
        3.0e-2
        * torch.ones(1, row * col, dtype=torch.float64)
    ).to(device)

    # Every CV fold and the final full-pool fit start from the same design.
    torch.manual_seed(0)
    k_n = (
        0.1e-4
        + 9.9e-4 * torch.rand(1, row * col, dtype=torch.float64)
    ).to(device) / beta**2
    if use_unilateral:
        k_d = (
            1.0e-4
            * torch.ones(1, row * col, dtype=torch.float64)
            / beta**2)
    else:
        k_d = torch.zeros(1, row * col, dtype=torch.float64)
    k_d = k_d.to(device)
    k_cubic = (
        fixed_cubic_coefficient
        * torch.ones(1, row * col, dtype=torch.float64)
        if use_cubic
        else torch.zeros(1, row * col, dtype=torch.float64)
    ).to(device)
    # Number of inter-cell couplers (vertical plus horizontal edges).
    edge_count = (row - 1) * col + row * (col - 1)
    k_c = (
        0.1e-4
        + 9.9e-4
        * torch.rand(1, edge_count, dtype=torch.float64)
        / beta**2
    ).to(device)
    c_n = (
        0.3e-4
        * torch.ones(1, row * col, dtype=torch.float64)
        / beta
    ).to(device)
    c_c = (
        0.0e-4
        * torch.ones(1, edge_count, dtype=torch.float64)
        / beta
    ).to(device)
    c_g = 1.0e-4
    # Reference stiffnesses stored as Coupling buffers (not used in the
    # dynamics).
    kn_o = (
        1.0e-3
        * torch.ones(1, row * col, dtype=torch.float64)
        / beta**2
    ).to(device)
    kc_o = (
        1.0e-3
        * torch.rand(1, edge_count, dtype=torch.float64)
        / beta**2
    ).to(device)

    # Assemble the lattice: topology, physical parameters, time-step cell,
    # source, fixed corner nodes and output probes.
    coup_mat = generateCoupMat.init_coupling_mat(
        row, col, k_c, k_n, m_c, m_n, c_c, c_n)
    coupling = metarnn.Coupling(
        domain_shape, row, col, m_n, m_c, k_n, k_c,
        c_n, c_c, coup_mat, kn_o, kc_o, k_d,
        c_g=c_g, k_cubic=k_cubic)
    cell = metarnn.WaveCell(
        dt,
        coupling,
        switch_temperature=switch_temperature,
        use_surrogate_gradient=(
            use_surrogate_gradient and use_unilateral),
        use_nonlinear=use_nonlinear,
        nonlinearity_mode=model_mode,
        state_limit=state_limit)
    node_ids = metarnn.default_node_ids(row, col)
    srcpos = node_ids["source"]
    source = metarnn.WaveSource(srcpos, dt)
    fixed_positions = node_ids["fixed"]
    if len(node_ids["probes"]) != NUM_CLASSES:
        raise ValueError(
            f"Geometry provides {len(node_ids['probes'])} probes, but "
            f"the dataset has {NUM_CLASSES} classes")
    probes = [
        metarnn.WaveIntensityProbe(probe_id)
        for probe_id in node_ids["probes"]
    ]
    print(
        f"Grid {row}x{col}: source={srcpos}, "
        f"fixed={list(fixed_positions)}, "
        f"probes={list(node_ids['probes'])}")

    model = metarnn.WaveRNN(
        cell, source, dt, fixed_positions, srcpos, probes
    ).double().to(device)
    model.cell.coup.k_d.requires_grad_(use_unilateral)

    initial_stability = project_stiffness_with_compatibility(model)
    print(
        "Initial stability: dt^2*lambda_max={:.4f}, scale={:.6f}"
        .format(
            initial_stability["dt2_lambda_max"],
            initial_stability["projection_scale"]))
    if use_nonlinear:
        initial_nonlinear_stats = nonlinear_coefficient_summary(model)
        print(
            "Initial {}: min={:.6e}, mean={:.6e}, max={:.6e}"
            .format(
                initial_nonlinear_stats["name"],
                initial_nonlinear_stats["minimum"],
                initial_nonlinear_stats["mean"],
                initial_nonlinear_stats["maximum"]))

    # Adam with separate parameter groups: linear springs (k_n, k_c) and,
    # in unilateral mode, k_d.
    optimizer_parameter_groups = [{
        "params": [model.cell.coup.k_n, model.cell.coup.k_c],
        "lr": 0.6e-5,
    }]
    if use_unilateral:
        optimizer_parameter_groups.append({
            "params": [model.cell.coup.k_d],
            "lr": 1.0e-6,
        })

    optimizer = torch.optim.Adam(
        optimizer_parameter_groups, betas=(0.9, 0.99))
    # Learning-rate schedule: epochs 1-10 decay by 0.98 per epoch; epoch 11
    # restarts at 1e-6 and then decays by 0.98 per epoch. LambdaLR is used
    # so that the CV folds and the final full-pool fit record and lock the
    # same learning-rate trajectory.
    def lr_lambda(epoch):
        # LambdaLR evaluates epoch 0 during initialization. Keep the
        # optimizer's configured base LR for the first training epoch.
        if epoch == 0:
            return 1.0
        if epoch <= 9:
            return 0.98 ** epoch
        # scheduler.step() runs after each epoch, so this value is used at
        # the start of epoch 11. The first ten training epochs therefore
        # use the 0.98-per-epoch start-of-epoch LRs. The base
        # linear LR is 6e-6, so 1/6 starts epoch 11 at exactly 1e-6.
        return (1.0 / 6.0) * (0.98 ** (epoch - 10))

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lr_lambda)
    return model, optimizer, scheduler


# ---------------------------------------------------------------------------
# Stratified k-fold cross-validation on the training pool
# ---------------------------------------------------------------------------
print('Cross-validation double-precision training has started ...')

k_folds = 5
folds_to_run = (1,2,3,4,5)
if not folds_to_run or any(
        fold < 1 or fold > k_folds for fold in folds_to_run):
    raise ValueError(
        f"folds_to_run must contain fold numbers from 1 to {k_folds}")

skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
fold_results = {} 
dummy_x = np.zeros(len(y))
labels_numpy = y.numpy()
# Global seeds. The fold split and the DataLoader order use their own
# fixed seeds below.
np.random.seed(42)
torch.manual_seed(0)

# For each fold: train with early stopping, restore the selected
# checkpoint, evaluate it and save its model and stiffnesses.
for fold, (train_ids, validation_ids) in enumerate(
        skf.split(dummy_x, labels_numpy)):
    current_fold = fold + 1
    if current_fold not in folds_to_run:
        continue
    print(f'\n--------------------------------')
    print(f'FOLD {current_fold}/{k_folds}')
    print(f'--------------------------------')
    
    train_evaluation_dataset = Subset(
        fullDataset, train_ids.tolist())
    validation_dataset = Subset(
        fullDataset, validation_ids.tolist())

    # Shuffle training samples while preserving deterministic stratified folds.
    train_generator = torch.Generator()
    train_generator.manual_seed(42 + current_fold * 1000)
    train_loader = torch.utils.data.DataLoader(
        train_evaluation_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=train_generator)
    train_evaluation_loader = torch.utils.data.DataLoader(
        train_evaluation_dataset, batch_size=160, shuffle=False)
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset, batch_size=160, shuffle=False)

    model, optimizer, scheduler = create_model_optimizer_scheduler()
    
    loss_iter = []
    time_pass = []
    lr_list = []
    epoch_start_lr_schedule = {}
    initial_validation_metrics = evaluate(
        model, validation_loader, 0, current_fold, "Validation")
    best_validation_ce_loss = initial_validation_metrics["ce_loss"]
    best_validation_acc = initial_validation_metrics["accuracy"]
    best_early_stopping_ce_loss = initial_validation_metrics["ce_loss"]
    best_epoch = 0
    best_model_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0

    for epoch in range(1, max_epoch+1):
        starttime = datetime.datetime.now()
        epoch_start_lr_schedule[epoch] = tuple(
            group["lr"] for group in optimizer.param_groups)
        try:
            optimization_train_loss = train(
                model, train_loader, optimizer, epoch, max_epoch,
                current_fold, lr_list, loss_iter)
            train_metrics = evaluate(
                model, train_evaluation_loader,
                epoch, current_fold, "Train")
            validation_metrics = evaluate(
                model, validation_loader, epoch, current_fold, "Validation")
        except FloatingPointError as error:
            failure_message = (
                f"Numerical failure in fold {current_fold}, epoch {epoch}: "
                f"{error}. Aborting without evaluating or saving this run.\n")
            print(failure_message)
            with open(train_log_filepath, "a") as f:
                f.write(failure_message)
            raise RuntimeError(failure_message) from error

        validation_ce_loss = validation_metrics["ce_loss"]
        # LambdaLR is stepped once per epoch.
        scheduler.step()
        endtime = datetime.datetime.now()
        time_pass.append(endtime - starttime)
        linear_parameters_trainable = (
            model.cell.coup.k_n.requires_grad
            or model.cell.coup.k_c.requires_grad)
        current_linear_lr = (
            optimizer.param_groups[0]["lr"]
            if linear_parameters_trainable
            else 0.0)
        current_nonlinear_lr = (
            optimizer.param_groups[1]["lr"]
            if len(optimizer.param_groups) > 1
            else 0.0)
        nonlinear_stats = nonlinear_coefficient_summary(model)

        # Train loss and Validation loss below are directly comparable:
        # both are post-epoch CE values evaluated with the same fixed model.
        epoch_summary = (
            f"Epoch summary (Fold {current_fold}, Epoch {epoch}): "
            f"Train loss={train_metrics['ce_loss']:.6f}, "
            f"Validation loss={validation_ce_loss:.6f}, "
            f"Validation accuracy={validation_metrics['accuracy']:.2f}%, "
            f"Validation minimum class accuracy="
            f"{validation_metrics['minimum_class_accuracy']:.2f}%, "
            f"Lr(kn,kc)={current_linear_lr:.3e}, "
            f"Lr(kd)={current_nonlinear_lr:.3e}\n")
        stiffness_summary = (
            f"Nonlinear coefficient summary "
            f"(Fold {current_fold}, Epoch {epoch}): "
            f"{nonlinear_stats['name']}(min/mean/max)="
            f"{nonlinear_stats['minimum']:.6e}/"
            f"{nonlinear_stats['mean']:.6e}/"
            f"{nonlinear_stats['maximum']:.6e}\n")
        detailed_epoch_summary = (
            f"Epoch loss comparison "
            f"(Fold {current_fold}, Epoch {epoch}): "
            f"OptimizationTrainTotal={optimization_train_loss:.6f}, "
            f"TrainCE={train_metrics['ce_loss']:.6f}, "
            f"TrainMargin={train_metrics['margin_loss']:.6f}, "
            f"TrainTotal={train_metrics['total_loss']:.6f}, "
            f"TrainAccuracy={train_metrics['accuracy']:.2f}%, "
            f"ValidationCE={validation_metrics['ce_loss']:.6f}, "
            f"ValidationMargin={validation_metrics['margin_loss']:.6f}, "
            f"ValidationTotal={validation_metrics['total_loss']:.6f}, "
            f"ValidationAccuracy={validation_metrics['accuracy']:.2f}%\n")
        print(
            epoch_summary + detailed_epoch_summary + stiffness_summary,
            end="")
        with open(train_log_filepath, "a") as f:
            f.write(
                epoch_summary + detailed_epoch_summary
                + stiffness_summary)

        validation_accuracy = validation_metrics["accuracy"]
        accuracy_improved = (
            validation_accuracy
            > best_validation_acc
            + model_selection_accuracy_tolerance)
        accuracy_tied = (
            abs(validation_accuracy - best_validation_acc)
            <= model_selection_accuracy_tolerance)
        ce_improved = (
            validation_ce_loss
            < best_validation_ce_loss - early_stopping_min_delta)
        is_better_checkpoint = (
            accuracy_improved
            or (accuracy_tied and ce_improved))

        if is_better_checkpoint:
            if accuracy_improved:
                selection_reason = "higher validation accuracy"
            else:
                selection_reason = (
                    "equal validation accuracy and lower validation CE")

            best_validation_ce_loss = validation_ce_loss
            best_validation_acc = validation_accuracy
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())

            checkpoint_message = (
                f"Best checkpoint updated "
                f"(Fold {current_fold}, Epoch {epoch}): "
                f"reason={selection_reason}, "
                f"Validation accuracy={best_validation_acc:.2f}%, "
                f"Validation CE={best_validation_ce_loss:.6f}\n")
            print(checkpoint_message, end="")
            with open(train_log_filepath, "a") as f:
                f.write(checkpoint_message)

        # Early stopping follows validation CE rather than the discrete
        # accuracy-first checkpoint rule, so that training is not stopped
        # during a temporary accuracy plateau.
        if validation_ce_loss < (
                best_early_stopping_ce_loss
                - early_stopping_min_delta):
            best_early_stopping_ce_loss = validation_ce_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= early_stopping_patience:
            stop_message = (
                f"Early stopping in fold {current_fold} at epoch {epoch}; "
                f"validation CE did not improve beyond "
                f"{best_early_stopping_ce_loss:.6f}; "
                f"restoring selected epoch {best_epoch} with "
                f"validation accuracy "
                f"{best_validation_acc:.2f}% and validation CE "
                f"{best_validation_ce_loss:.6f}.\n")
            print(stop_message, end="")
            with open(train_log_filepath, "a") as f:
                f.write(stop_message)
            break

    # Restore the selected checkpoint of this fold.
    model.load_state_dict(best_model_state)
    best_train_metrics = evaluate(
        model, train_evaluation_loader,
        best_epoch, current_fold, "Best train")
    best_validation_metrics = evaluate(
        model, validation_loader,
        best_epoch, current_fold, "Best validation")

    best_checkpoint_summary = (
        f"Best-checkpoint comparison "
        f"(Fold {current_fold}, Epoch {best_epoch}): "
        f"Train(CE/Margin/Total/Accuracy)="
        f"{best_train_metrics['ce_loss']:.6f}/"
        f"{best_train_metrics['margin_loss']:.6f}/"
        f"{best_train_metrics['total_loss']:.6f}/"
        f"{best_train_metrics['accuracy']:.2f}%, "
        f"Validation(CE/Margin/Total/Accuracy)="
        f"{best_validation_metrics['ce_loss']:.6f}/"
        f"{best_validation_metrics['margin_loss']:.6f}/"
        f"{best_validation_metrics['total_loss']:.6f}/"
        f"{best_validation_metrics['accuracy']:.2f}%\n")
    print(best_checkpoint_summary, end="")
    with open(train_log_filepath, "a") as f:
        f.write(best_checkpoint_summary)

    best_model_filename = (
        f"model_{model_mode}_double_{time_now}_"
        f"fold{current_fold}_best.pkl")
    torch.save(model, "./model/" + best_model_filename)

    kc = model.cell.coup.k_c
    kn = model.cell.coup.k_n
    kd = model.cell.coup.k_d
    k_cubic = model.cell.coup.k_cubic
    np.savetxt(
        f'parameter_save/parameter_kc_{model_mode}_{time_now}_'
        f'fold{current_fold}_best.txt',
        kc.detach().cpu().numpy())
    np.savetxt(
        f'parameter_save/parameter_kn_{model_mode}_{time_now}_'
        f'fold{current_fold}_best.txt',
        kn.detach().cpu().numpy())
    np.savetxt(
        f'parameter_save/parameter_kd_{model_mode}_{time_now}_'
        f'fold{current_fold}_best.txt',
        kd.detach().cpu().numpy())
    if use_cubic:
        np.savetxt(
            f'parameter_save/parameter_kcubic_{model_mode}_{time_now}_'
            f'fold{current_fold}_fixed.txt',
            k_cubic.detach().cpu().numpy())
    print(
        f"--> [Fold {current_fold}] Restored epoch {best_epoch}; "
        f"validation accuracy="
        f"{best_validation_metrics['accuracy']:.2f}%, "
        f"minimum class accuracy="
        f"{best_validation_metrics['minimum_class_accuracy']:.2f}%\n")

    fold_results[current_fold] = {
        "best_epoch": best_epoch,
        "train": best_train_metrics,
        "validation": best_validation_metrics,
        "epoch_start_lr_schedule": epoch_start_lr_schedule,
    }

    # Release fold-specific objects before constructing the next model.
    del (
        model, optimizer, scheduler, train_loader,
        train_generator, train_evaluation_loader, validation_loader,
        train_evaluation_dataset, validation_dataset,
        best_model_state, initial_validation_metrics, best_train_metrics,
        best_validation_metrics,
        kc, kn, kd, k_cubic)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ---------------------------------------------------------------------------
# Cross-validation summary
# ---------------------------------------------------------------------------
print('\n======================================')
print(
    f'CROSS-VALIDATION SUMMARY FOR FOLDS {folds_to_run} '
    f'({model_mode.upper()}, FLOAT64)')
print('======================================')
for f, result in fold_results.items():
    validation = result["validation"]
    print(
        f"Fold {f} (epoch {result['best_epoch']}): "
        f"Validation CE={validation['ce_loss']:.6f}, "
        f"Margin={validation['margin_loss']:.6f}, "
        f"Total={validation['total_loss']:.6f}, "
        f"Accuracy={validation['accuracy']:.2f}%")

num_completed_folds = len(fold_results)
if num_completed_folds:
    average_validation_accuracy = sum(
        result["validation"]["accuracy"]
        for result in fold_results.values()) / num_completed_folds
    average_validation_ce = sum(
        result["validation"]["ce_loss"]
        for result in fold_results.values()) / num_completed_folds
    average_validation_margin = sum(
        result["validation"]["margin_loss"]
        for result in fold_results.values()) / num_completed_folds
    average_validation_total = sum(
        result["validation"]["total_loss"]
        for result in fold_results.values()) / num_completed_folds
    print(
        f"Average metrics over {num_completed_folds} fold model(s): "
        f"Validation CE={average_validation_ce:.6f}, "
        f"Margin={average_validation_margin:.6f}, "
        f"Total={average_validation_total:.6f}, "
        f"Accuracy={average_validation_accuracy:.2f}%")
print('======================================')

# A final model is fitted only after all CV folds have completed. Its epoch
# budget and learning-rate path are locked from CV alone; the complete
# training pool is then used without a validation split or early stopping.
if num_completed_folds == len(folds_to_run):
    # Lock the epoch budget (rounded median of the CV best epochs) and the
    # per-epoch learning rates (median over folds).
    cv_best_epochs = np.asarray([
        result["best_epoch"] for result in fold_results.values()
    ], dtype=np.float64)
    final_training_epochs = max(
        1, int(np.floor(np.median(cv_best_epochs) + 0.5)))

    locked_lr_schedule = {}
    for epoch in range(1, final_training_epochs + 1):
        fold_learning_rates = [
            result["epoch_start_lr_schedule"][epoch]
            for result in fold_results.values()
            if epoch in result["epoch_start_lr_schedule"]
        ]
        if not fold_learning_rates:
            raise RuntimeError(
                f"No CV learning-rate record is available for epoch {epoch}")
        group_count = len(fold_learning_rates[0])
        if any(len(rates) != group_count for rates in fold_learning_rates):
            raise RuntimeError(
                "CV folds have inconsistent optimizer parameter groups")
        locked_lr_schedule[epoch] = tuple(
            float(np.median([
                rates[group_index]
                for rates in fold_learning_rates
            ]))
            for group_index in range(group_count)
        )

    final_protocol_message = (
        "\n======================================\n"
        "FINAL FULL-TRAINING-POOL FIT\n"
        "======================================\n"
        f"CV best epochs={cv_best_epochs.astype(int).tolist()}; "
        f"locked final epoch budget={final_training_epochs} "
        "(rounded median); learning rates are per-epoch CV medians.\n"
        "The final fit uses all training samples and does not use validation "
        "or feedback from the independent-test split.\n")
    print(final_protocol_message, end="")
    with open(train_log_filepath, "a") as f:
        f.write(final_protocol_message)

    # Retrain a fresh model on the full training pool with the locked
    # schedule (no validation split, no early stopping).
    final_train_generator = torch.Generator()
    final_train_generator.manual_seed(424242)
    final_train_loader = torch.utils.data.DataLoader(
        fullDataset,
        batch_size=batch_size,
        shuffle=True,
        generator=final_train_generator)
    final_train_evaluation_loader = torch.utils.data.DataLoader(
        fullDataset, batch_size=160, shuffle=False)

    final_model, final_optimizer, final_scheduler = (
        create_model_optimizer_scheduler())
    del final_scheduler
    final_lr_list = []
    final_loss_iter = []
    final_train_metrics = None

    for epoch in range(1, final_training_epochs + 1):
        epoch_learning_rates = locked_lr_schedule[epoch]
        if len(epoch_learning_rates) != len(final_optimizer.param_groups):
            raise RuntimeError(
                "Locked CV learning-rate groups do not match final optimizer")
        for parameter_group, locked_lr in zip(
                final_optimizer.param_groups, epoch_learning_rates):
            parameter_group["lr"] = locked_lr

        try:
            final_optimization_loss = train(
                final_model,
                final_train_loader,
                final_optimizer,
                epoch,
                final_training_epochs,
                0,
                final_lr_list,
                final_loss_iter)
            final_train_metrics = evaluate(
                final_model,
                final_train_evaluation_loader,
                epoch,
                0,
                "Final train")
        except FloatingPointError as error:
            failure_message = (
                f"Numerical failure in final full-pool fit, epoch {epoch}: "
                f"{error}. The independent test split was not evaluated.\n")
            print(failure_message, end="")
            with open(train_log_filepath, "a") as f:
                f.write(failure_message)
            raise RuntimeError(failure_message) from error

        final_epoch_message = (
            f"Final epoch summary (Epoch {epoch}): "
            f"OptimizationTrainTotal={final_optimization_loss:.6f}, "
            f"TrainCE={final_train_metrics['ce_loss']:.6f}, "
            f"TrainMargin={final_train_metrics['margin_loss']:.6f}, "
            f"TrainTotal={final_train_metrics['total_loss']:.6f}, "
            f"TrainAccuracy={final_train_metrics['accuracy']:.2f}%, "
            f"LockedLRs={list(epoch_learning_rates)}\n")
        print(final_epoch_message, end="")
        with open(train_log_filepath, "a") as f:
            f.write(final_epoch_message)

    # Do not load the held-out split until CV locking and final full-pool
    # retraining are complete. Its results never feed back into the model.
    independent_test_dataset, _ = load_classification_split(
        dataset_root / "test",
        "independent test",
        EXPECTED_SAMPLES_PER_CLASS["test"])
    require_disjoint_waveforms(
        fullDataset, independent_test_dataset,
        "training", "independent test")
    independent_test_loader = torch.utils.data.DataLoader(
        independent_test_dataset, batch_size=400, shuffle=False)
    final_independent_test_metrics = evaluate(
        final_model,
        independent_test_loader,
        final_training_epochs,
        0,
        "Final independent test")
    final_summary = (
        f"Final locked model (Epoch {final_training_epochs}): "
        f"Train(CE/Margin/Total/Accuracy)="
        f"{final_train_metrics['ce_loss']:.6f}/"
        f"{final_train_metrics['margin_loss']:.6f}/"
        f"{final_train_metrics['total_loss']:.6f}/"
        f"{final_train_metrics['accuracy']:.2f}%, "
        f"IndependentTest(CE/Margin/Total/Accuracy)="
        f"{final_independent_test_metrics['ce_loss']:.6f}/"
        f"{final_independent_test_metrics['margin_loss']:.6f}/"
        f"{final_independent_test_metrics['total_loss']:.6f}/"
        f"{final_independent_test_metrics['accuracy']:.2f}%\n")
    print(final_summary, end="")
    with open(train_log_filepath, "a") as f:
        f.write(final_summary)

    # Save the final model and its trained stiffnesses.
    final_model_filename = (
        f"model_{model_mode}_double_{time_now}_final.pkl")
    torch.save(final_model, "./model/" + final_model_filename)
    np.savetxt(
        f"parameter_save/parameter_kc_{model_mode}_{time_now}_final.txt",
        final_model.cell.coup.k_c.detach().cpu().numpy())
    np.savetxt(
        f"parameter_save/parameter_kn_{model_mode}_{time_now}_final.txt",
        final_model.cell.coup.k_n.detach().cpu().numpy())
    np.savetxt(
        f"parameter_save/parameter_kd_{model_mode}_{time_now}_final.txt",
        final_model.cell.coup.k_d.detach().cpu().numpy())
    if use_cubic:
        np.savetxt(
            f"parameter_save/parameter_kcubic_{model_mode}_{time_now}_"
            "final_fixed.txt",
            final_model.cell.coup.k_cubic.detach().cpu().numpy())
else:
    print(
        "Final full-training-pool fit skipped: all five CV folds must be "
        "completed in the same run.")
