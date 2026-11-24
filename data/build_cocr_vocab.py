"""Build the CoCR functional-requirement vocabulary (paper App. A.2.2).

Only the TRAIN split is used to build the category->function prior.
Outputs (data/cocr/):
    func_vocab.json        K functional nodes: name, description, keyword regexes
    func_text_emb.npy      (K, 768) RoBERTa mean-pooled embedding of each node description
    cat_func_prior.npy     (485, K) P(f | object category), estimated from train intents
    func_targets_{split}.npy (N, K) soft functional target per annotation (aligned with {split}.json)
        train: keyword multi-hot (Eq.14) + category prior of the GT target
        val/test: keyword multi-hot only (never uses target annotations)

Usage:
    python data/build_cocr_vocab.py --data_root [DATA_ROOT]
"""
import argparse
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# (name, description, keyword regexes). Regexes are matched on the lower-cased intent.
FUNCTIONS = [
    ("placement surface", "a flat surface to place or put things down on",
     [r"\bplac(e|ing)\b", r"\bput(ting)?\b", r"\bset (it |them |my )?down\b", r"\blay(ing)?\b", r"\bsurface\b", r"\bonto\b"]),
    ("work and study support", "a place to work or study on tasks",
     [r"\bwork(ing|s)?\b", r"\bstud(y|ying)\b", r"\bworkspace\b", r"\btasks?\b", r"\bprojects?\b", r"\bhomework\b"]),
    ("dining support", "support for eating meals and snacks",
     [r"\beat(ing)?\b", r"\bmeals?\b", r"\bdinner\b", r"\blunch\b", r"\bbreakfast\b", r"\bdin(e|ing)\b", r"\bsnacks?\b"]),
    ("storage", "storing and keeping belongings",
     [r"\bstor(e|ing|age)\b", r"\bkeep(ing)?\b", r"\bstash\b", r"\bbelongings\b", r"\bessentials\b", r"\bsupplies\b"]),
    ("organization", "organizing and arranging items neatly",
     [r"\borgani[sz](e|ed|ing)\b", r"\barrang(e|ing)\b", r"\btid(y|ily)\b", r"\bneatly\b", r"\bsort\b", r"\bin order\b"]),
    ("display", "displaying or showcasing items",
     [r"\bdisplay(ing)?\b", r"\bshowcas(e|ing)\b", r"\bshow off\b", r"\bexhibit\b"]),
    ("carrying", "carrying and transporting things",
     [r"\bcarry(ing)?\b", r"\bpack(ing)?\b", r"\btransport\b", r"\btrip\b", r"\btravel(ing|ling)?\b", r"\bbring\b"]),
    ("waste disposal", "throwing away and disposing of waste",
     [r"\bdispos(e|al|ing)\b", r"\bdiscard(ing)?\b", r"\bthrow(ing)? (away|out)\b", r"\btrash\b", r"\bwaste\b", r"\bgarbage\b",
      r"\bunwanted\b", r"\bwrappers?\b", r"\brecycl(e|ing|able)\b", r"\bget rid of\b"]),
    ("seating", "somewhere to sit down",
     [r"\bsit(ting)?\b", r"\bseat(ed|ing)?\b"]),
    ("rest and relaxation", "resting and relaxing comfortably",
     [r"\brest(ing)?\b", r"\brelax(ing|ation)?\b", r"\bnap\b", r"\blounge\b", r"\bunwind\b", r"\bcomfortabl[ey]\b", r"\bcomfort\b"]),
    ("body support", "supporting the body, back, feet or head",
     [r"\bsupport\b", r"\bback\b", r"\bfeet\b", r"\blean\b", r"\blegs?\b", r"\bhead\b", r"\bposture\b", r"\bneck\b"]),
    ("sleeping", "sleeping and lying down at night",
     [r"\bsleep(ing)?\b", r"\bnight\b", r"\blie down\b", r"\bbedtime\b"]),
    ("writing", "writing, jotting notes and drawing",
     [r"\bwrit(e|ing)\b", r"\bjot\b", r"\bnotes?\b", r"\bdraw(ing)?\b", r"\bsketch\b", r"\bideas\b", r"\breminders?\b"]),
    ("presentation and meeting", "presenting to colleagues in meetings",
     [r"\bpresent(ation|ations|ing)?\b", r"\bmeetings?\b", r"\bconference\b", r"\bcolleagues\b", r"\bteam\b",
      r"\bdiscussions?\b", r"\bclients\b", r"\bbrainstorm(ing)?\b"]),
    ("visual display screen", "watching shows and viewing a screen",
     [r"\bwatch(ing)?\b", r"\btv\b", r"\bscreen\b", r"\bmovies?\b", r"\bshows\b", r"\bnews\b", r"\bview(ing)?\b", r"\bmonitor\b"]),
    ("illumination", "providing light to see in the dark",
     [r"\blight(ing)?\b", r"\billuminat(e|ion|ing)\b", r"\bbright\b", r"\bdark\b", r"\blamp\b"]),
    ("cleaning", "cleaning and wiping dirty things",
     [r"\bclean(ing)?\b", r"\bwip(e|ing)\b", r"\bscrub\b", r"\bspills?\b", r"\bdirty\b", r"\bdust\b", r"\bdishes\b", r"\bmess\b"]),
    ("bathing", "bathing, showering and soaking the body",
     [r"\bbath(e|ing)?\b", r"\bshower(ing)?\b", r"\bsoak(ing)?\b"]),
    ("cooking", "cooking and preparing food",
     [r"\bcook(ing)?\b", r"\bprepar(e|ing)\b", r"\bboil\b", r"\bvegetables\b", r"\bingredients\b", r"\brecipe\b", r"\bheat (up )?(my )?food\b"]),
    ("drinking and liquid containment", "holding and drinking water, coffee or tea",
     [r"\bdrink(s|ing)?\b", r"\bwater\b", r"\bcoffee\b", r"\btea\b", r"\bbeverages?\b", r"\bpour\b", r"\bhydrat(e|ed|ion)\b"]),
    ("wearing", "wearing clothes and outfits",
     [r"\bwear(ing)?\b", r"\boutfits?\b", r"\bdress(ed)?\b", r"\bput on\b", r"\boutside\b"]),
    ("decoration", "decorating the room and making it aesthetic",
     [r"\bdecorat(e|ion|ing)\b", r"\baesthetic\b", r"\batmosphere\b", r"\bbeautif(y|ul)\b", r"\bart(work)?\b",
      r"\bappreciat(e|ing)\b", r"\bmemories\b", r"\bambiance\b"]),
    ("hanging", "hanging things up",
     [r"\bhang(ing)?\b", r"\bhook\b"]),
    ("privacy and concealment", "blocking sunlight and keeping privacy",
     [r"\bprivacy\b", r"\bblock(ing)?\b", r"\bcover(ing)?\b", r"\bhide\b", r"\bsunlight\b", r"\bshade\b", r"\bconceal\b"]),
    ("heating", "keeping warm and heating the room",
     [r"\bwarm(th)?\b", r"\bheat(ing)?\b", r"\bcold\b", r"\bwinter\b", r"\btemperature\b", r"\bcozy\b"]),
    ("cooling and ventilation", "cooling the air and ventilating",
     [r"\bcool(ing)?\b", r"\bair\b", r"\bbreeze\b", r"\bventilat(e|ion)\b", r"\bhot\b", r"\bsmoke\b", r"\bfumes\b"]),
    ("food preservation", "keeping food fresh and chilled",
     [r"\bfresh\b", r"\bleftovers?\b", r"\bgroceries\b", r"\brefrigerat(e|ed)\b", r"\bchill(ed)?\b", r"\bperishable\b", r"\bfood\b"]),
    ("hand washing and hygiene", "washing hands and maintaining hygiene",
     [r"\bwash(ing)?\b", r"\bhands\b", r"\bhygiene\b", r"\bsanitiz(e|ing)\b", r"\bsoap\b", r"\brinse\b"]),
    ("toilet use", "using the toilet for bodily needs",
     [r"\btoilet\b", r"\bbodily\b", r"\brelieve myself\b", r"\brestroom\b", r"\bbathroom\b", r"\bpee\b", r"\bnature calls\b"]),
    ("drying", "drying off hands and body",
     [r"\bdry(ing)?\b", r"\bdry off\b", r"\btowels?\b"]),
    ("grooming and appearance", "checking appearance and grooming",
     [r"\bappearance\b", r"\bmirror\b", r"\bgroom(ing)?\b", r"\bhair\b", r"\bmakeup\b", r"\bface\b", r"\bshav(e|ing)\b",
      r"\bteeth\b", r"\blook\b", r"\bcheck\b"]),
    ("computing and digital input", "typing and using a computer",
     [r"\btyp(e|ing)\b", r"\bcomputer\b", r"\bemails?\b", r"\bdata\b", r"\bdigital\b", r"\binput\b", r"\blaptop\b",
      r"\bonline\b", r"\bsoftware\b", r"\bbrows(e|ing)\b", r"\bgames?\b"]),
    ("communication", "making calls and communicating",
     [r"\bcalls?\b", r"\bphone\b", r"\bcommunicat(e|ion)\b", r"\bcontact\b", r"\btalk\b", r"\btelephone\b"]),
    ("printing and copying", "printing, copying and scanning documents",
     [r"\bprint(ing|out|outs)?\b", r"\bcop(y|ies|ying)\b", r"\bscan(ning)?\b"]),
    ("reading", "reading books and learning",
     [r"\bread(ing)?\b", r"\bbooks?\b", r"\bknowledge\b", r"\blibrary\b", r"\bnovels?\b", r"\blearn(ing)?\b"]),
    ("documents and paperwork", "handling documents, files and papers",
     [r"\bdocuments?\b", r"\bfiles?\b", r"\bpaperwork\b", r"\bpapers?\b", r"\breports?\b", r"\bfolders?\b", r"\bmaterials\b"]),
    ("time keeping", "checking the time and keeping schedule",
     [r"\btime\b", r"\bclock\b", r"\bschedule\b", r"\bpunctual\b", r"\blate\b", r"\bhours\b"]),
    ("laundry", "collecting and washing laundry",
     [r"\blaundry\b", r"\bdirty clothes\b", r"\bhamper\b", r"\bclothes\b"]),
    ("switch and control", "turning devices on or off and adjusting settings",
     [r"\bturn(ing)? (on|off)\b", r"\bswitch\b", r"\bcontrol\b", r"\badjust(ing)?\b"]),
    ("greenery and nature", "bringing plants and nature indoors",
     [r"\bplants?\b", r"\bnature\b", r"\bgreen(ery)?\b", r"\bflowers?\b", r"\bgarden(ing)?\b"]),
]


def match_functions(text, compiled):
    text = re.sub(r"[^a-z0-9' ]", " ", text.lower())
    text = re.sub(r"\s+", " ", text)
    return np.array([1.0 if any(p.search(text) for p in pats) else 0.0 for pats in compiled], dtype=np.float32)


def category_index(name, type2class):
    name = name.lower().strip()
    cands = [name, name.rstrip('s'), name[:-2] if name.endswith('es') else name,
             name.replace('clothing', 'clothes'), name.replace('cushion', 'couch cushions'),
             name.split(' ')[-1], name.split(' ')[-1].rstrip('s')]
    for c in cands:
        if c in type2class:
            return type2class[c]
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--out_dir', default='data/cocr')
    parser.add_argument('--roberta', default='roberta-base')
    parser.add_argument('--prior_smooth', type=float, default=1.0)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    from data.model_util_scannet import ScannetDatasetConfig
    DC = ScannetDatasetConfig(485)
    type2class = DC.type2class
    K = len(FUNCTIONS)
    compiled = [[re.compile(p) for p in pats] for _, _, pats in FUNCTIONS]

    # ---- category -> function prior from TRAIN only ----
    train = json.load(open(os.path.join(args.data_root, 'intention_sentence', 'train.json')))
    counts = np.zeros((485, K), np.float64)
    n_cat = np.zeros(485, np.float64)
    kw_train = np.stack([match_functions(a['utterance'], compiled) for a in train])
    cat_ids = []
    unmapped = {}
    for a, y in zip(train, kw_train):
        c = category_index(a['target'], type2class)
        cat_ids.append(c)
        if c is None:
            unmapped[a['target']] = unmapped.get(a['target'], 0) + 1
            continue
        counts[c] += y
        n_cat[c] += 1
    glob = kw_train.mean(0)
    prior = (counts + args.prior_smooth * glob[None]) / (n_cat[:, None] + args.prior_smooth)
    np.save(os.path.join(args.out_dir, 'cat_func_prior.npy'), prior.astype(np.float32))

    # ---- soft targets ----
    def to_target(y, p=None):
        t = y.copy()
        if p is not None:
            t = t + p / max(p.sum(), 1e-6) * max(y.sum(), 1.0)
        if t.sum() <= 0:
            t = np.ones(K, np.float32)
        return (t / t.sum()).astype(np.float32)

    tr_t = np.stack([to_target(y, prior[c] if c is not None else None) for y, c in zip(kw_train, cat_ids)])
    np.save(os.path.join(args.out_dir, 'func_targets_train.npy'), tr_t)
    stats = {'train_keyword_coverage': float((kw_train.sum(1) > 0).mean()),
             'train_mean_funcs': float(kw_train.sum(1).mean()),
             'node_coverage': {FUNCTIONS[j][0]: int(kw_train[:, j].sum()) for j in range(K)},
             'unmapped_targets': unmapped}
    for split in ['val', 'test']:
        annos = json.load(open(os.path.join(args.data_root, 'intention_sentence', f'{split}.json')))
        kw = np.stack([match_functions(a['utterance'], compiled) for a in annos])
        np.save(os.path.join(args.out_dir, f'func_targets_{split}.npy'), np.stack([to_target(y) for y in kw]))
        stats[f'{split}_keyword_coverage'] = float((kw.sum(1) > 0).mean())

    # ---- text embeddings of node descriptions ----
    import torch
    from transformers import RobertaModel, RobertaTokenizerFast
    tok = RobertaTokenizerFast.from_pretrained(args.roberta)
    enc = RobertaModel.from_pretrained(args.roberta).eval()
    texts = [f"{n}: {d}" for n, d, _ in FUNCTIONS]
    with torch.no_grad():
        t = tok(texts, padding=True, return_tensors='pt')
        h = enc(**t).last_hidden_state
        m = t.attention_mask.unsqueeze(-1).float()
        emb = (h * m).sum(1) / m.sum(1)
    np.save(os.path.join(args.out_dir, 'func_text_emb.npy'), emb.numpy().astype(np.float32))

    json.dump({'functions': [{'name': n, 'description': d, 'patterns': p} for n, d, p in FUNCTIONS],
               'stats': stats}, open(os.path.join(args.out_dir, 'func_vocab.json'), 'w'), indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == '__main__':
    main()
