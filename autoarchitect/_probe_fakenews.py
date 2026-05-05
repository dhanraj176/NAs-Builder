import pyarrow as pa, json
from pathlib import Path
from collections import Counter

base = Path('datasets/hf_cache/GonzaloA___fake_news')
print('Folder exists:', base.exists())
arrows = list(base.rglob('*.arrow'))
print('Arrow files:', [str(f.relative_to(base)) for f in arrows])

for f in base.rglob('dataset_info.json'):
    d = json.load(open(f))
    print('Features:', list(d.get('features', {}).keys()))
    for sname, sinfo in d.get('splits', {}).items():
        n = sinfo.get('num_examples', '?')
        print(f'  split {sname}: {n} examples')

if arrows:
    r = pa.ipc.open_stream(str(arrows[0]))
    print('\nSchema:', r.schema.names)
    batch = r.read_next_batch()
    print('First batch rows:', batch.num_rows)
    for col in r.schema.names:
        val = batch.column(col)[0].as_py()
        if isinstance(val, str):
            print(f'  {col}: str = "{val[:100]}"')
        elif isinstance(val, (bytes, bytearray)):
            print(f'  {col}: bytes len={len(val)}')
        else:
            print(f'  {col}: {type(val).__name__} = {val}')

    # Count total rows and label distribution
    all_labels = []
    for arrow_f in arrows:
        r2 = pa.ipc.open_stream(str(arrow_f))
        for b in r2:
            all_labels.extend(b.column('label').to_pylist())
    print(f'\nTotal rows: {len(all_labels)}')
    print('Label dist:', Counter(all_labels))
