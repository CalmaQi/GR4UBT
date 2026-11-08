"""Training budgets for the selected Stage-I recipe."""
from paths import DATA as SOURCE, ROOT, RUNS as OUT

def budgets(ds, phase):
    if phase == 'full':
        return dict(dense_epochs=120, dense_patience=16, joint_epochs=24, joint_patience=8, max_joint_rows=None)
    return dict(dense_epochs=50 if ds == 'lastfm' else 80, dense_patience=12,
                joint_epochs=8 if ds == 'lastfm' else 16, joint_patience=4 if ds == 'lastfm' else 6,
                max_joint_rows=131072 if ds == 'lastfm' else None)
