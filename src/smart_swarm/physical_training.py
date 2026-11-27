"""Train official motor policies from verified native prefix readers and train-only statistics."""
from bisect import bisect_right
from pathlib import Path
import time
from .episode import atomic, load, sha
from .physical_motor_dataset import PhysicalMotorDataset, CAMERA_KEYS


class NativeTrainingData(PhysicalMotorDataset):
    def __getitem__(self, index):
        row = super().__getitem__(index)
        import torch.nn.functional as functional
        for camera in CAMERA_KEYS:
            row[camera] = functional.interpolate(row[camera][None], size=(128,128), mode='bilinear', align_corners=False)[0]
        return row

    def identity(self, index):
        place = bisect_right(self._stops, index)
        previous = self._stops[place-1] if place else 0
        segment = self._episodes[place]
        return {'export': str(segment._native.root), 'native_dataset_row': segment.start + index - previous,
                'episode_prefix': place, 'prefix_local_row': index-previous}


def training_statistics(reader):
    import torch
    from torch.utils.data import DataLoader
    sums, squares, counts = {}, {}, {}
    for batch in DataLoader(reader, batch_size=8):
        values = {'observation.state': batch['observation.state'], 'action': batch['action'][:,0]}
        values.update({key: batch[key].permute(0,2,3,1).reshape(-1,3) for key in CAMERA_KEYS})
        for key, value in values.items():
            data = value.double()
            sums[key] = sums.get(key, 0) + data.sum(0)
            squares[key] = squares.get(key, 0) + data.square().sum(0)
            counts[key] = counts.get(key, 0) + len(data)
    stats = {}
    for key in sums:
        mean = sums[key]/counts[key]
        std = (squares[key]/counts[key]-mean.square()).clamp_min(1e-8).sqrt()
        shape = (3,1,1) if key in CAMERA_KEYS else (6,)
        stats[key] = {'mean': mean.float().reshape(shape), 'std': std.float().reshape(shape)}
    return stats


def policy_for(model, stats, *, technical=False, weights=None, backbone=None):
    from lerobot.configs.types import PolicyFeature, FeatureType
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors
    inputs = {'observation.state': PolicyFeature(FeatureType.STATE, (6,)),
              **{key: PolicyFeature(FeatureType.VISUAL, (3,128,128)) for key in CAMERA_KEYS}}
    outputs = {'action': PolicyFeature(FeatureType.ACTION, (6,))}
    config = ACTConfig(input_features=inputs, output_features=outputs, device='cpu', chunk_size=3 if technical else 24,
                       n_action_steps=1 if technical else 6, dim_model=32 if technical else 128,
                       n_heads=4, dim_feedforward=64 if technical else 512, n_encoder_layers=1 if technical else 2,
                       n_decoder_layers=1, n_vae_encoder_layers=1, pretrained_backbone_weights=None)
    if model == 'smolvla':
        from lerobot.configs import PreTrainedConfig
        if weights is None or backbone is None or not (Path(backbone)/'tokenizer.json').is_file():
            raise ValueError('SmolVLA requires the verified local initialization and tokenizer')
        config = PreTrainedConfig.from_pretrained(str(weights))
        config.input_features, config.output_features, config.device = inputs, outputs, 'cpu'
        config.vlm_model_name = str(backbone)
    elif model != 'act':
        raise ValueError('Unsupported native motor model')
    policy_type = get_policy_class(model)
    policy = policy_type.from_pretrained(str(weights), config=config) if weights else policy_type(config)
    pre, post = make_pre_post_processors(config, dataset_stats=stats)
    return policy.float(), pre, post


def native_train(corpus, output, *, model='act', technical=False, updates=1, wall_seconds=90, weights=None, backbone=None):
    import torch
    torch.set_num_threads(2)
    torch.manual_seed(3)
    if not 1 <= updates <= 5000 or not 1 <= wall_seconds <= 7200:
        raise ValueError('Native training exceeds the original update or wall cap')
    started = time.monotonic()
    record = load(corpus)
    split = 'development' if technical else 'train'
    chunk = 3 if technical else 50 if model=='smolvla' else 24
    reader = NativeTrainingData(record, split, chunk, technical=technical)
    if not len(reader):
        raise ValueError('INSUFFICIENT_DATA. No admitted native training samples')
    from .lab_jobs import motor_eligibility
    if not technical and not motor_eligibility(record)['eligible']:
        raise ValueError('INSUFFICIENT_DATA. Native training requires independent train and validation groups')
    output.mkdir(parents=True, exist_ok=False)
    stats = training_statistics(reader)
    policy, pre, post = policy_for(model, stats, technical=technical, weights=weights, backbone=backbone)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-5)
    exposure = []
    losses = optimize(policy, pre, reader, optimizer, updates, started + wall_seconds, exposure)
    checkpoint = output / 'pretrained_model'
    policy.save_pretrained(checkpoint)
    pre.save_pretrained(checkpoint)
    post.save_pretrained(checkpoint)
    evaluation = reader if technical else NativeTrainingData(record, 'validation', chunk, technical=False)
    proof = reload_evaluate(checkpoint, evaluation)
    receipt = {'status': 'trained', 'model': model, 'source': record['source'], 'technical_only': technical,
               'run_mode': 'native_replay_fixture' if technical else 'hardware_recordings', 'corpus_sha256': sha(corpus),
               'optimizer_updates': len(losses), 'losses': losses, 'sample_ids': exposure, 'split': split,
               'normalization_source': 'training prefix rows only', 'ordered_cameras': list(CAMERA_KEYS),
               'image_profile': 'native-two-rgb-128-v1', 'checkpoint': str(checkpoint), 'evaluation': proof,
               'closed_loop_manipulations': 0, 'elapsed_s': time.monotonic()-started}
    atomic(output / 'run.json', receipt)
    return receipt


def optimize(policy, pre, reader, optimizer, updates, deadline, exposure):
    import torch
    from torch.utils.data import default_collate
    losses = []
    policy.train()
    for update in range(updates):
        if time.monotonic() >= deadline:
            raise TimeoutError('Native motor fit reached its declared wall cap')
        index = update % len(reader)
        batch = default_collate([reader[index]])
        optimizer.zero_grad()
        loss, _ = policy(pre(batch))
        if not torch.isfinite(loss):
            raise ValueError('Native motor fit produced nonfinite loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.)
        optimizer.step()
        losses.append(float(loss.detach()))
        exposure.append(reader.identity(index))
    return losses


def reload_evaluate(checkpoint, reader):
    import torch
    from torch.utils.data import default_collate
    from lerobot.configs import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors
    config = PreTrainedConfig.from_pretrained(str(checkpoint))
    policy = get_policy_class(config.type).from_pretrained(str(checkpoint), config=config).eval()
    pre, post = make_pre_post_processors(config, pretrained_path=str(checkpoint))
    row = reader[0]
    batch = default_collate([row])
    policy.reset()
    with torch.inference_mode():
        action = post(policy.select_action(pre(batch)))
    if action.shape != (1,6) or not torch.isfinite(action).all():
        raise ValueError('Reloaded native policy emitted invalid decoded commands')
    return {'scope': 'offline native sample after checkpoint reload', 'sample_id': reader.identity(0),
            'decoded_action': action[0].tolist(), 'mean_absolute_native_action_error': float((action[0]-row['action'][0]).abs().mean()),
            'task_outcome': 'unmeasured', 'controller_execution': 'not_requested'}
