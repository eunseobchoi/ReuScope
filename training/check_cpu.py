"""Check CLI help and input/output/path boundaries using only the standard library."""
import argparse
import ast
from contextlib import redirect_stderr
import importlib.util
import io
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parent


def check_confidence_preflight(tree):
    # Execute the actual main with only its dataframe boundary faked; stop before outputs.
    class Frame:
        def __init__(self, columns, rows):
            self.columns, self.rows = columns, rows

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, columns):
            return Frame(columns, [[row[self.columns.index(c)] for c in columns] for row in self.rows])

        def itertuples(self, index=False, name=None):
            assert index is False and name is None
            return iter(self.rows)

    class PreflightPassed(Exception):
        pass

    class StopBeforeOutputs:
        def __truediv__(self, output):
            raise PreflightPassed

    valid = Frame(['image_path', 'Ulcer', 'Normal'], [[r'training\Normal\a.jpg', 0, 1]] * 2)
    inputs = ('train.csv', 'orig.csv', 'dedup.csv')
    frames = dict.fromkeys(inputs, valid)
    reads = []

    def read(path):
        reads.append(path)
        return frames[path]

    nodes = [node for node in tree.body if
             isinstance(node, ast.FunctionDef) and node.name in ('build_parser', 'main') or
             isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'CV2024_CLASSES' for t in node.targets)]
    namespace = {'argparse': argparse, 'ROOT': ROOT, 'OUT': StopBeforeOutputs(),
                 'require_directory': lambda name: ROOT,
                 'pd': SimpleNamespace(read_csv=read)}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<confidence preflight>', 'exec'), namespace)
    parser = namespace['build_parser']()

    def run(error=None, **overrides):
        args = parser.parse_args(['--train_csv', inputs[0], '--orig_val_csv', inputs[1],
                                  '--dedup_val_csv', inputs[2], '--output', 'unused/metrics.json',
                                  '--batch', '2', '--no_batch128_assert', '--seeds', '0', '4294967295'])
        vars(args).update(overrides)
        reads.clear()
        stderr = io.StringIO()
        try:
            with redirect_stderr(stderr):
                namespace['main'](parser, args)
        except SystemExit as exc:
            assert error and exc.code == 2 and error in stderr.getvalue(), stderr.getvalue()
        except PreflightPassed:
            assert error is None, error
        else:
            raise AssertionError('main did not stop at the preflight boundary')

    for field in ('epochs', 'batch'):
        for value in (0, -1):
            run('must be positive', **{field: value})
            assert not reads
    for seeds in ([], [0, 0], [-1], [2**32]):
        run('unique uint32', seeds=seeds)
        assert not reads
    for input_name, path in zip(('train_csv', 'orig_val_csv', 'dedup_val_csv'), inputs):
        for values in ([0, 0], [1, 1], [0.5, 0.5], [-1, 2], [float('nan'), 1],
                       [float('inf'), 0], [-float('inf'), 1], ['1', 0]):
            frames[path] = Frame(valid.columns, [['a.jpg', *values]] * 2)
            run(f'{input_name}: row 1 requires finite binary')
        for image_path in ('', ' \t ', None, float('nan'), 123):
            frames[path] = Frame(valid.columns, [[image_path, 0, 1]] * 2)
            run('requires a nonblank image_path string')
        for frame in (Frame(valid.columns, []), Frame(['image_path', 'unknown'], [['a.jpg', 1]] * 2),
                      Frame(['Normal'], [[1]] * 2)):
            frames[path] = frame
            run('expected nonempty rows')
        frames[path] = valid
    run('row count must be at least batch', batch=3)
    run()
    assert reads == list(inputs)
    frames['orig.list'] = valid
    run(orig_val_csv='orig.list')
    assert reads == [inputs[0], 'orig.list', inputs[2]]
    assert valid.rows == [[r'training\Normal\a.jpg', 0, 1]] * 2
    print('PASS confidence main preflight: invalid inputs rejected before outputs/model; valid subset/duplicates and CSV parsing regardless of suffix')

    def stop_before_training(*args):
        raise PreflightPassed

    with TemporaryDirectory() as directory:
        output_root = Path(directory)
        namespace.update(OUT=output_root, Path=Path,
                         re=SimpleNamespace(search=stop_before_training))
        final = output_root / 'new/metrics.json'
        paths = [final, final.with_name('metrics.json.partial'), final.with_name('metrics.json.tmp'),
                 output_root / 'predictions.csv', output_root / 'trainable.pt']
        exports = dict(seeds=[0], output=str(final), prediction_output=str(paths[3]),
                       trainable_checkpoint=str(paths[4]))
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'prior result')
            run('Refusing to replace prior output', **exports)
            assert path.read_bytes() == b'prior result'
            path.unlink()
        for path in paths[:3]:
            run('paths must be distinct', **(exports | {'prediction_output': str(path)}))
            run('paths must be distinct', **(exports | {'trainable_checkpoint': str(path)}))
        run('paths must be distinct', **(exports | {'trainable_checkpoint': str(paths[3])}))
        alias = output_root / 'alias'
        alias.symlink_to(final.parent, target_is_directory=True)
        run('paths must be distinct', **(exports | {'prediction_output': str(alias / final.name)}))
        alias.unlink()
        run(**exports)
        assert not any(path.is_file() for path in output_root.rglob('*'))
    print('PASS confidence output preflight: prior outputs and colliding paths rejected; fresh paths accepted')


def check_official_loader():
    image = MagicMock()
    transforms = SimpleNamespace(Compose=lambda steps: steps, Resize=lambda size: ('resize', size),
                                 ToTensor=lambda: 'tensor', Normalize=lambda *args: ('normalize', args))
    modules = {'PIL': SimpleNamespace(Image=image),
               'torch.utils.data': SimpleNamespace(Dataset=object),
               'torchvision': SimpleNamespace(transforms=transforms)}
    with patch.dict(sys.modules, modules), patch.object(sys, 'path', [str(ROOT / 'src'), *sys.path]), \
            patch.dict(os.environ, {}, clear=True), TemporaryDirectory() as directory:
        import dataset_official as alias
        from utils import dataset_kvasir_official as loader

        assert alias is loader
        assert loader.OFFICIAL_CLASSES == ['Angiectasia', 'Blood', 'Erosion', 'Erythematous',
                                           'Foreign Bodies', 'Ileo-cecal valve', 'Lymphangiectasia',
                                           'Normal', 'Pylorus', 'Reduced Mucosal View', 'Ulcer']
        assert loader.EVAL_TRANSFORM == [('resize', (224, 224)), 'tensor',
                                         ('normalize', ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))]

        def rejected(error, **kwargs):
            try:
                loader.KvasirCapsuleOfficial(**kwargs)
            except ValueError as exc:
                assert error in str(exc), str(exc)
            else:
                raise AssertionError(f'Accepted invalid official split: {kwargs}')

        rejected('split must', split='../split_0')
        rejected('Set KVASIR_ROOT')
        root = Path(directory)
        alias.set_data_paths(data_root=root / 'missing', splits_dir=root)
        rejected('existing directories')
        alias.set_data_paths(data_root=root)
        assert loader.DATA_ROOT == root and alias.SPLITS_DIR == root
        split = root / 'split_0.csv'
        for contents, error in [('', 'unique filename and label'),
                                ('filename\na.jpg\n', 'unique filename and label'),
                                ('filename,label,label\na.jpg,Normal,Ulcer\n', 'unique filename and label'),
                                ('filename,label\na.jpg\n', 'Malformed'),
                                ('filename,label\na.jpg,\n', 'Malformed'),
                                ('filename,label\na.jpg,Normal,extra\n', 'Malformed'),
                                ('filename,label\n', 'No supported images'),
                                ('filename,label\na.jpg,unsupported\n', 'No supported images')]:
            split.write_text(contents)
            rejected(error)
        for filename in ('', ' ', '.', '..', '/outside.jpg', '../outside.jpg', 'dir/a.jpg',
                         r'..\outside.jpg', r'C:\outside.jpg', 'C:outside.jpg', 'a\x00.jpg'):
            split.write_text(f'filename,label\n{filename},Normal\n')
            rejected('filename without directories')
        rows = [('z.jpg', 'Ulcer'), ('a.jpg', 'Normal'), ('z.jpg', 'Ulcer')]
        rows += [('frame.jpg', name) for name in reversed(loader.OFFICIAL_CLASSES)]
        contents = 'filename,label,metadata\n' + ''.join(f'{fn},{label},unused\n' for fn, label in rows)
        contents += 'ignored.jpg,unsupported,unused\n'
        for name in ('split_0', 'split_1'):
            (root / f'{name}.csv').write_text(contents)
            dataset = loader.KvasirCapsuleOfficial(split=name, transform=None)
            assert dataset.items == [(str(root / loader.LABEL_TO_FOLDER[label] / fn),
                                      loader.CLASS_TO_IDX[label], label) for fn, label in rows]
            assert len(dataset) == len(rows)
        converted = image.open.return_value.__enter__.return_value.convert.return_value
        assert dataset[0] == (converted, loader.CLASS_TO_IDX['Ulcer'], 0)
        image.open.assert_called_once_with(str(root / 'ulcer/z.jpg'))
        image.open.return_value.__enter__.return_value.convert.assert_called_once_with('RGB')
        image.open.return_value.__exit__.assert_called_once_with(None, None, None)
        transform = MagicMock(return_value='transformed')
        dataset.transform = transform
        assert dataset[1] == ('transformed', loader.CLASS_TO_IDX['Normal'], 1)
        transform.assert_called_once_with(converted)
        image.open.return_value.__enter__.return_value.convert.side_effect = OSError('bad image')
        try:
            dataset[0]
        except OSError:
            assert image.open.return_value.__exit__.call_args.args[0] is OSError
        else:
            raise AssertionError('Swallowed image conversion failure')
    print('PASS official loader: alias identity, roots, CSV/filename validation, order/duplicates, classes and image lifecycle')


def main():
    if sys.flags.optimize:
        raise SystemExit('Run without -O: interface checks require assertions')
    check_official_loader()
    check_confidence_preflight(ast.parse((ROOT / 'src/confidence.py').read_text()))
    for script in ('primary.py', 'confidence.py', 'cross_model.py'):
        tree = ast.parse((ROOT / 'src' / script).read_text())
        parser_node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'build_parser')
        namespace = {'argparse': argparse}
        exec(compile(ast.Module(body=[parser_node], type_ignores=[]), script, 'exec'), namespace)
        parser = namespace['build_parser']()
        required = ['--train_csv', 'train.csv', '--orig_val_csv', 'original.csv',
                    '--dedup_val_csv', 'evaluation.csv', '--output', 'metrics.json']
        if script == 'cross_model.py':
            required += ['--backbone', 'resnet50']
        assert parser.parse_args(required).orig_val_csv == 'original.csv'
        for flag in ('--train_csv', '--orig_val_csv', '--dedup_val_csv'):
            index = required.index(flag)
            with redirect_stderr(io.StringIO()):
                try:
                    parser.parse_args(required[:index] + required[index + 2:])
                except SystemExit as exc:
                    assert exc.code == 2
                else:
                    raise AssertionError(f'{script} accepted missing {flag}')
    print('PASS all trainers require explicit training and evaluation lists')

    spec = importlib.util.spec_from_file_location('path_resolution', ROOT / 'src/utils/path_resolution.py')
    paths = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(paths)
    for value, expected in [(r'training\Normal\KVASIR\frame.jpg', Path('/cv/training/Normal/KVASIR/frame.jpg')),
                            ('kvasir_capsule/labelled_images/ulcer/frame.jpg', Path('/kv/ulcer/frame.jpg')),
                            ('labelled_images/ulcer/frame.jpg', Path('/kv/ulcer/frame.jpg')),
                            ('kvasir_capsule_split_1/ulcer/frame.jpg', Path('/kv/ulcer/frame.jpg')),
                            ('kvasir_capsule_split_1/nested/ulcer/frame.jpg', Path('/kv/ulcer/frame.jpg')),
                            ('/absolute/frame.jpg', Path('/absolute/frame.jpg'))]:
        assert paths.resolve_cv2024_or_kvasir_path(value, cv2024_root=Path('/cv'), kvasir_data_root=Path('/kv')) == expected
    for value in ('kvasir_capsule_split_1/', 'kvasir_capsule_split_1/frame.jpg',
                  'kvasir_capsule_split_1/ulcer/', 'kvasir_capsule_split_1//frame.jpg', '', ' ', None, 123):
        try:
            paths.resolve_cv2024_or_kvasir_path(value, cv2024_root=Path('/cv'), kvasir_data_root=Path('/kv'))
        except ValueError:
            pass
        else:
            raise AssertionError(f'Accepted malformed image path: {value!r}')
    for prefix in ('kvasir_capsule_split_1/', 'kvasir_capsule/labelled_images/', 'labelled_images/'):
        for suffix in ('', ' ', '/outside.jpg', '../outside.jpg', 'ulcer/../../outside.jpg',
                       'ulcer/', 'ulcer/.', r'..\outside.jpg', r'C:\outside.jpg', 'C:outside.jpg',
                       'ulcer/a\x00.jpg'):
            try:
                paths.resolve_cv2024_or_kvasir_path(prefix + suffix, cv2024_root=Path('/cv'),
                                                   kvasir_data_root=Path('/kv'))
            except ValueError:
                pass
            else:
                raise AssertionError(f'Accepted malformed Kvasir image path: {prefix + suffix!r}')
    for value in ('training/frame.jpg', 'labelled_images/ulcer/frame.jpg'):
        try:
            paths.resolve_cv2024_or_kvasir_path(value, cv2024_root=None, kvasir_data_root=None)
        except ValueError:
            pass
        else:
            raise AssertionError('Accepted relative image path without an explicit root')
    for value in (None, '', str(ROOT / 'missing-directory'), str(Path(__file__))):
        with patch.dict(os.environ, {}, clear=True):
            if value is not None:
                os.environ['CV2024_ROOT'] = value
            try:
                paths.require_directory('CV2024_ROOT')
            except ValueError as exc:
                assert 'CV2024_ROOT' in str(exc)
            else:
                raise AssertionError('Accepted missing or invalid directory')
    with patch.dict(os.environ, {'CV2024_ROOT': str(ROOT)}):
        assert paths.require_directory('CV2024_ROOT') == ROOT
    print('PASS path normalization and explicit directory validation')

    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
    for key in ('CAPSULE_ROOT', 'CAPSULE_ARTIFACT_ROOT', 'CV2024_ROOT', 'KVASIR_ROOT',
                'KVASIR_SPLITS_DIR', 'DINOV2_REPO', 'TORCH_HOME', 'PYTHONPATH'):
        env.pop(key, None)
    commands = [(['src/primary.py'], '--train_csv'),
                (['src/confidence.py'], '--prediction_output'),
                (['src/cross_model.py'], '--backbone'),
                (['crossmodel.py'], 'summarize'),
                (['crossmodel.py', 'plan'], '--resnet50-weights'),
                (['crossmodel.py', 'run'], '--variant'),
                (['crossmodel.py', 'summarize'], '--runs')]
    for command, expected in commands:
        result = subprocess.run([sys.executable, '-S', str(ROOT / command[0]), *command[1:], '--help'],
                                env=env, cwd=ROOT.parent, text=True, capture_output=True, timeout=60)
        assert result.returncode == 0 and expected in result.stdout, (command, result.stdout, result.stderr)
        print('PASS', ' '.join(command), '--help')


if __name__ == '__main__':
    main()
