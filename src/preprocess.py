"""Preprocessing pipeline: raw twcs.csv -> AmazonHelp grounding_pool / eval_pool.

Stages:
  A load_and_filter_brand  - load raw CSV, find brand-authored tweet ids
  B reconstruct_threads    - walk in_response_to_tweet_id / response_tweet_id
                              to assemble full conversation threads
  C clean_and_filter       - strip boilerplate, filter English, dedup,
                              tag is_triage, drop threads with no brand reply
  [checkpoint]             - manual review of is_triage tagging (required)
  D subsample              - not yet implemented (pending checkpoint sign-off)
  E split_pools            - not yet implemented (pending checkpoint sign-off)

Run: python src/preprocess.py --input data/twcs/twcs.csv --brand AmazonHelp
"""
import argparse
import json
import random
import re
from collections import deque
from pathlib import Path

import pandas as pd

try:
    from langdetect import detect, DetectorFactory, LangDetectException
    DetectorFactory.seed = 42
except ImportError:
    detect = None
    LangDetectException = Exception


MENTION_RE = re.compile(r'@\w+')
# Matches an optional ": " directly before a URL or phone number, plus the URL/phone itself
# (greedy \S+ for the URL, since a lookahead-based stop-at-punctuation approach breaks on
# shortened domains like "t.co" -- the internal "." looks identical to a trailing sentence
# period). Trailing sentence punctuation glued onto the greedy match is trimmed back off in
# _strip_redirect_target below and preserved in the output, instead of being deleted with the
# URL -- that's what previously truncated or spliced sentences together around removed links.
_URL_PATTERN = r'https?://\S+'
_PHONE_PATTERN = r'\b1?[-.]?\(?\d{3}\)?[-.\s]\d{3}[-.]\d{4}\b'
REDIRECT_TARGET_RE = re.compile(rf'(?::\s*)?(?P<target>{_URL_PATTERN}|{_PHONE_PATTERN})')
SIGNATURE_RE = re.compile(r'\^[A-Za-z]{1,3}\s*$')
SPACE_BEFORE_PUNCT_RE = re.compile(r'\s+([.,!?;:])')
WHITESPACE_RE = re.compile(r'\s+')


def _strip_redirect_target(match: 're.Match') -> str:
    target = match.group('target')
    trailing = ''
    while target and target[-1] in '.,:;!?':
        trailing = target[-1] + trailing
        target = target[:-1]
    return ' ' + trailing

TRIAGE_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r'\bdm\b',
        r'\bdirect message',
        r'\bprivate message',
        r'\bsend (us|you) a (dm|private|direct)',
        r'\bphone or chat\b',
        r'\bcall us\b',
        r'\bgive (us|you) a call\b',
        r'\bcontact us\b',
        r'\bcontact (our|the) support team\b',
        r'\breach (us|out)\b',
        r'\bget in touch\b',
        # (no separate literal-digit phone pattern here: clean_text has phone numbers stripped
        # by REDIRECT_TARGET_RE before tagging runs, so a digit-shaped pattern would never
        # match; the phrasing patterns above already catch these replies via wording.)
        # imperative redirect: "please/kindly <action> ... <target noun>" within a short window,
        # e.g. "please share your details here:", "kindly reach out to our support team here:"
        r'\b(please|kindly)\b[^.?!]{0,40}\b(share|provide|fill|contact|reach|report|escalate|check|send)\b'
        r'[^.?!]{0,40}\b(details|information|form|summary|correspondence|support team)\b',
        r'\byou can do so here\b',
        r'\bget in touch using the link\b',
        r'\bfill (in |out )?(this|the)? ?form\b',
    ]
]

MAX_COMPONENT_SIZE = 200


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ''
    t = MENTION_RE.sub('', text)
    t = REDIRECT_TARGET_RE.sub(_strip_redirect_target, t)
    t = SIGNATURE_RE.sub('', t)
    t = SPACE_BEFORE_PUNCT_RE.sub(r'\1', t)
    t = WHITESPACE_RE.sub(' ', t).strip()
    t = t.rstrip(' :;,')
    if t and t[-1] not in '.!?':
        t += '.'
    return t


def is_triage_reply(clean: str) -> bool:
    return any(p.search(clean) for p in TRIAGE_PATTERNS)


def _safe_detect(text: str) -> str:
    text = text.strip()
    if len(text) < 3 or detect is None:
        return 'unknown'
    try:
        return detect(text)
    except LangDetectException:
        return 'unknown'


def load_and_filter_brand(input_path: Path, brand: str):
    print(f"[Stage A] Loading {input_path} ...")
    df = pd.read_csv(input_path, dtype=str, keep_default_na=False, na_values=[])
    print(f"[Stage A] Loaded {len(df):,} rows")
    seed_ids = df.loc[df['author_id'] == brand, 'tweet_id'].tolist()
    print(f"[Stage A] {len(seed_ids):,} rows authored by {brand}")
    return df, seed_ids


def build_indices(df: pd.DataFrame):
    id_to_row = {}
    children_map = {}
    for row in df.itertuples(index=False):
        id_to_row[row.tweet_id] = row
        if row.response_tweet_id:
            for child in row.response_tweet_id.split(','):
                child = child.strip()
                if child:
                    children_map.setdefault(row.tweet_id, []).append(child)
    return id_to_row, children_map


def reconstruct_threads(df: pd.DataFrame, seed_ids, max_component_size=MAX_COMPONENT_SIZE):
    print("[Stage B] Building id index ...")
    id_to_row, children_map = build_indices(df)

    visited_global = set()
    thread_rows = []
    n_components = 0
    n_oversized_skipped = 0

    for seed in seed_ids:
        if seed in visited_global:
            continue

        component = set()
        queue = deque([seed])
        while queue:
            tid = queue.popleft()
            if tid in component:
                continue
            component.add(tid)
            row = id_to_row.get(tid)
            if row is None:
                continue
            parent = row.in_response_to_tweet_id
            if parent and parent not in component:
                queue.append(parent)
            for child in children_map.get(tid, []):
                if child not in component:
                    queue.append(child)
            if len(component) > max_component_size:
                break

        if len(component) > max_component_size:
            n_oversized_skipped += 1
            visited_global.update(component)
            continue

        visited_global.update(component)
        n_components += 1

        rows = [id_to_row[t] for t in component if t in id_to_row]
        roots = [r for r in rows if not r.in_response_to_tweet_id]
        root = min(roots, key=lambda r: r.created_at) if roots else min(rows, key=lambda r: r.created_at)
        thread_id = root.tweet_id

        for r in rows:
            thread_rows.append({
                'thread_id': thread_id,
                'tweet_id': r.tweet_id,
                'author_id': r.author_id,
                'inbound': r.inbound,
                'created_at': r.created_at,
                'text': r.text,
            })

    print(f"[Stage B] {n_components:,} threads reconstructed "
          f"({n_oversized_skipped:,} oversized components skipped, cap={max_component_size})")

    out = pd.DataFrame(thread_rows)
    out['created_at_parsed'] = pd.to_datetime(
        out['created_at'], format='%a %b %d %H:%M:%S %z %Y', errors='coerce'
    )
    n_bad_dates = out['created_at_parsed'].isna().sum()
    if n_bad_dates:
        print(f"[Stage B] Warning: {n_bad_dates:,} rows had unparseable created_at")

    out = out.sort_values(['thread_id', 'created_at_parsed'])
    out['turn_index'] = out.groupby('thread_id').cumcount()
    out['created_at'] = out['created_at_parsed'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    out = out.drop(columns=['created_at_parsed']).reset_index(drop=True)
    return out


def clean_and_filter(threads_df: pd.DataFrame, brand: str):
    print("[Stage C] Cleaning text ...")
    threads_df = threads_df.copy()
    threads_df['clean_text'] = threads_df['text'].map(clean_text)

    before = len(threads_df)
    threads_df = threads_df.sort_values(['thread_id', 'turn_index'])
    threads_df = threads_df.drop_duplicates(subset=['author_id', 'clean_text'], keep='first')
    print(f"[Stage C] Dropped {before - len(threads_df):,} exact-duplicate tweets")

    threads_df = threads_df.sort_values(['thread_id', 'turn_index'])
    threads_df['turn_index'] = threads_df.groupby('thread_id').cumcount()

    is_brand = threads_df['author_id'] == brand
    threads_df['is_triage'] = False
    threads_df.loc[is_brand, 'is_triage'] = threads_df.loc[is_brand, 'clean_text'].map(is_triage_reply)
    print(f"[Stage C] Tagged {threads_df.loc[is_brand, 'is_triage'].sum():,}/{is_brand.sum():,} "
          f"brand replies as is_triage")

    print("[Stage C] Detecting language of brand replies (this may take a while) ...")
    # Per-turn detection: a thread is dropped if ANY brand reply is confidently non-English.
    # Replies too short to classify come back 'unknown' and don't block the thread (avoids
    # false-dropping short replies like "Thanks!"). Detecting on the whole thread's
    # concatenated brand text (the earlier approach) let code-switched threads slip through,
    # since an English-majority blob still classifies as 'en' even if one turn is in Spanish.
    brand_rows = threads_df[is_brand]
    turn_lang = brand_rows['clean_text'].map(_safe_detect)
    non_english_threads = set(brand_rows.loc[~turn_lang.isin(['en', 'unknown']), 'thread_id'])

    before_threads = threads_df['thread_id'].nunique()
    threads_df = threads_df[~threads_df['thread_id'].isin(non_english_threads)]
    print(f"[Stage C] Kept {threads_df['thread_id'].nunique():,}/{before_threads:,} threads after English filter")

    before_threads2 = threads_df['thread_id'].nunique()
    has_brand_reply = threads_df.groupby('thread_id')['author_id'].transform(lambda s: (s == brand).any())
    threads_df = threads_df[has_brand_reply]
    print(f"[Stage C] Kept {threads_df['thread_id'].nunique():,}/{before_threads2:,} threads "
          f"with a brand reply remaining after dedup/lang filtering")

    return threads_df.reset_index(drop=True)


def print_triage_checkpoint(threads_df: pd.DataFrame, brand: str, n=15):
    brand_rows = threads_df[threads_df['author_id'] == brand]
    print("\n" + "=" * 80)
    print("CHECKPOINT: review is_triage tagging before proceeding to Stage D")
    print("=" * 80)
    for label, val in [("TRUE", True), ("FALSE", False)]:
        subset = brand_rows[brand_rows['is_triage'] == val]
        sample = subset.sample(n=min(n, len(subset)), random_state=42)
        print(f"\n--- is_triage = {label} ({len(subset):,} total, showing {len(sample)}) ---")
        for _, row in sample.iterrows():
            print(f"  [{row['thread_id']}] {row['clean_text']}")
    print("\n" + "=" * 80)


def subsample(threads_df: pd.DataFrame, brand: str, n_threads: int, seed: int):
    print(f"[Stage D] Subsampling to {n_threads:,} threads (seed={seed}) ...")

    n_turns = threads_df.groupby('thread_id')['turn_index'].max() + 1
    n_turns.name = 'n_turns'

    brand_rows = threads_df[threads_df['author_id'] == brand].sort_values('turn_index')
    last_brand_triage = brand_rows.groupby('thread_id').tail(1).set_index('thread_id')['is_triage']

    summary = pd.DataFrame({'n_turns': n_turns})
    summary['is_triage'] = last_brand_triage.reindex(summary.index).fillna(False)
    summary['length_bucket'] = pd.qcut(summary['n_turns'], q=3, labels=['short', 'medium', 'long'], duplicates='drop')
    summary['stratum'] = summary['length_bucket'].astype(str) + '|' + summary['is_triage'].astype(str)

    if n_threads >= len(summary):
        print(f"[Stage D] Requested {n_threads:,} >= available {len(summary):,}; keeping all threads")
        sample = summary
    else:
        frac = n_threads / len(summary)
        picked = []
        for _, group in summary.groupby('stratum'):
            k = min(int(round(len(group) * frac)), len(group))
            picked.append(group.sample(n=k, random_state=seed))
        sample = pd.concat(picked)

        if len(sample) > n_threads:
            sample = sample.sample(n=n_threads, random_state=seed)
        elif len(sample) < n_threads:
            remaining = summary.drop(sample.index)
            top_up = remaining.sample(n=min(n_threads - len(sample), len(remaining)), random_state=seed)
            sample = pd.concat([sample, top_up])

    print(f"[Stage D] Selected {len(sample):,} threads. Stratum breakdown:")
    print(sample.groupby('stratum').size().to_string())

    subset = threads_df[threads_df['thread_id'].isin(sample.index)].copy()
    return subset.sort_values(['thread_id', 'turn_index']).reset_index(drop=True)


def build_thread_record(thread_id: str, g: pd.DataFrame, brand: str):
    """One canonical (thread_history, customer_message, reference_reply) example per thread:
    reference_reply = the thread's last brand turn; customer_message = the customer turn
    immediately preceding it; thread_history = everything before that. Matches the brief's
    Section 2 agent input schema and Section 5 golden-set fields.
    """
    g = g.sort_values('turn_index').reset_index(drop=True)
    brand_idxs = g.index[g['author_id'] == brand].tolist()
    if not brand_idxs:
        return None
    reply_idx = brand_idxs[-1]

    # customer turns whose text survives cleaning (drop @mention/URL-only tweets, e.g. screenshot replies)
    customer_idxs = g.index[
        (g['author_id'] != brand) & (g.index < reply_idx) & (g['clean_text'].str.strip() != '')
    ].tolist()
    if not customer_idxs:
        return None
    customer_idx = customer_idxs[-1]

    reference_reply = g.loc[reply_idx, 'clean_text']
    if not reference_reply.strip():
        return None

    history = [
        {
            'speaker': 'customer' if row['author_id'] != brand else 'brand',
            'text': row['clean_text'],
            'created_at': row['created_at'],
        }
        for _, row in g.iloc[:customer_idx].iterrows()
        if row['clean_text'].strip()
    ]

    return {
        'thread_id': thread_id,
        'brand': brand,
        'customer_message': g.loc[customer_idx, 'clean_text'],
        'thread_history': history,
        'reference_reply': g.loc[reply_idx, 'clean_text'],
        'is_triage': bool(g.loc[reply_idx, 'is_triage']),
        'n_turns': int(len(g)),
    }


def split_pools(subset_df: pd.DataFrame, brand: str, eval_frac: float, seed: int, processed_dir: Path):
    print(f"[Stage E] Splitting into grounding_pool / eval_pool (eval_frac={eval_frac}, seed={seed}) ...")

    thread_ids = sorted(subset_df['thread_id'].unique().tolist())
    rnd = random.Random(seed)
    rnd.shuffle(thread_ids)
    n_eval = round(len(thread_ids) * eval_frac)
    eval_ids = set(thread_ids[:n_eval])
    grounding_ids = set(thread_ids[n_eval:])

    overlap = grounding_ids & eval_ids
    assert not overlap, f"Overlap detected between pools: {overlap}"
    assert grounding_ids | eval_ids == set(thread_ids)

    records = {}
    n_no_customer_turn = 0
    for thread_id, g in subset_df.groupby('thread_id'):
        rec = build_thread_record(thread_id, g, brand)
        if rec is None:
            n_no_customer_turn += 1
            continue
        records[thread_id] = rec
    if n_no_customer_turn:
        print(f"[Stage E] Skipped {n_no_customer_turn:,} threads with no customer turn before the brand reply")

    def write_jsonl(path, ids):
        rows = [records[t] for t in ids if t in records]
        with open(path, 'w') as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        return len(rows)

    n_g = write_jsonl(processed_dir / 'grounding_pool.jsonl', grounding_ids)
    n_e = write_jsonl(processed_dir / 'eval_pool.jsonl', eval_ids)
    print(f"[Stage E] grounding_pool.jsonl: {n_g:,} threads | eval_pool.jsonl: {n_e:,} threads | overlap: {len(overlap)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('data/twcs/twcs.csv'))
    parser.add_argument('--brand', type=str, default='AmazonHelp')
    parser.add_argument('--interim-dir', type=Path, default=Path('data/interim'))
    parser.add_argument('--processed-dir', type=Path, default=Path('data/processed'))
    parser.add_argument('--force', action='store_true',
                         help='Force Stage C rebuild (reuses cached 02_threads_raw.parquet if present)')
    parser.add_argument('--force-ab', action='store_true', help='Also force Stage A/B rebuild from --input')
    parser.add_argument('--checkpoint-n', type=int, default=15)
    parser.add_argument('--stop-after-checkpoint', action='store_true',
                         help='Stop after Stage C for manual is_triage review (used during heuristic development)')
    parser.add_argument('--n-threads', type=int, default=4000)
    parser.add_argument('--eval-frac', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    args.interim_dir.mkdir(parents=True, exist_ok=True)
    args.processed_dir.mkdir(parents=True, exist_ok=True)
    clean_path = args.interim_dir / '03_threads_clean.parquet'
    raw_path = args.interim_dir / '02_threads_raw.parquet'

    if clean_path.exists() and not args.force and not args.force_ab:
        print(f"[Stage A-C] Using cached {clean_path} (pass --force to rebuild)")
        threads_df = pd.read_parquet(clean_path)
    else:
        if raw_path.exists() and not args.force_ab:
            print(f"[Stage A-B] Using cached {raw_path} (pass --force-ab to rebuild from --input)")
            threads_raw = pd.read_parquet(raw_path)
        else:
            df, seed_ids = load_and_filter_brand(args.input, args.brand)
            threads_raw = reconstruct_threads(df, seed_ids)
            threads_raw.to_parquet(raw_path, index=False)
        threads_df = clean_and_filter(threads_raw, args.brand)
        threads_df.to_parquet(clean_path, index=False)
        print(f"[Stage C] Wrote {clean_path}")

    print_triage_checkpoint(threads_df, args.brand, n=args.checkpoint_n)

    if args.stop_after_checkpoint:
        print("--stop-after-checkpoint set: not running Stage D/E.")
        return

    subset_df = subsample(threads_df, args.brand, args.n_threads, args.seed)
    subset_path = args.processed_dir / 'sample_subset.csv'
    subset_df.to_csv(subset_path, index=False)
    print(f"[Stage D] Wrote {subset_path}")

    split_pools(subset_df, args.brand, args.eval_frac, args.seed, args.processed_dir)


if __name__ == '__main__':
    main()
