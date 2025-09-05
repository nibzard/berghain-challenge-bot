import json
import glob


def estimate_freqs_corr(events_glob: str = 'game_logs/events_*.jsonl'):
    import json as _json
    enc = {'both': 0, 'y_only': 0, 'w_only': 0, 'neither': 0}
    y = w = both = n = 0
    for path in glob.glob(events_glob):
        try:
            with open(path, 'r') as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        j = _json.loads(line)
                    except Exception:
                        continue
                    d = j.get('data', {})
                    attrs = d.get('attributes', {})
                    yy = bool(attrs.get('young'))
                    ww = bool(attrs.get('well_dressed'))
                    if yy and ww:
                        enc['both'] += 1
                        both += 1
                    elif yy:
                        enc['y_only'] += 1
                    elif ww:
                        enc['w_only'] += 1
                    else:
                        enc['neither'] += 1
                    y += 1 if yy else 0
                    w += 1 if ww else 0
                    n += 1
        except Exception:
            continue
    if n == 0:
        return None
    p_y = y / n
    p_w = w / n
    p11 = both / n
    # Pearson corr for Bernoulli
    import math
    denom = math.sqrt(max(p_y*(1-p_y), 1e-12) * max(p_w*(1-p_w), 1e-12))
    rho = (p11 - p_y*p_w) / denom if denom > 0 else 0.0
    return {
        'p_young': p_y,
        'p_well_dressed': p_w,
        'p_both': p11,
        'p_neither': 1.0 - (p11 + (p_y - p11) + (p_w - p11)),
        'corr_young_well_dressed': rho,
        'counts': enc,
        'total': n
    }


if __name__ == '__main__':
    est = estimate_freqs_corr()
    if not est:
        print('No events found')
    else:
        print(json.dumps(est, indent=2))
