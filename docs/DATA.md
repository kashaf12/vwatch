# Data

**No video files are in this repository.** Everything under `eval/data/` (videos, truth files, manifest, caches) is gitignored and rebuilt locally from the sources below. What the repo does track: the builders (`src/vwatch/dataset/`), fixed id lists for benchmark runs (`eval/bench/*.txt`), and reports with charts (`eval/results/`).

How each task's videos are made is in [dataset-spec.md](dataset-spec.md). This page covers where the footage comes from, its license, and how to rebuild it.

| Task | Source | License | Redistributed? | Rebuild |
|---|---|---|---|---|
| T1 | Your own videos (the maintainer used Hindi YouTube Shorts from their own rendering pipeline) | yours | No | bring your own, see below |
| T2, T3, T4 | Procedurally rendered by this repo | MIT, like the code | No (rendered locally) | `uv run vw dataset build --root eval/data --scale full --seed 7 --tasks T2 T3 T4` |
| T1x | Wikimedia Commons speech videos | per file: CC BY, CC BY-SA, public domain | No (rebuilt from Commons) | `uv run python -m vwatch.dataset.t1x --root eval/data --seed 7` |
| T5 | TVBench subset | CC-BY-4.0 | No (rebuilt from Hugging Face) | `uv run python -m vwatch.dataset.tvbench --root eval/data --per-category 10 --seed 7` |

Builders are deterministic given `--seed`, add their items to `eval/data/manifest.jsonl`, and append attribution to `eval/data/SOURCES.md`. They need `ffmpeg`, and network access for T1x and T5.

Every builder replaces only its own lines in the manifest (under a file lock), so they can run in any order. T4 includes the real-Short question set only when that source and its `props.json` are present; otherwise it is built from procedural clips alone.

## T5: TVBench subset

- Dataset: TVBench, https://huggingface.co/datasets/FunAILab/TVBench
- License: CC-BY-4.0 (annotations/videos; the repo's `license_notice.txt` adds that part of the video content comes from Apache-2.0 source datasets: Perception Test, CLEVRER, STAR and others listed in its README).
- Attribution: Daniel Cores, Michael Dorkenwald, Manuel Mucientes, Cees G. M. Snoek, Yuki M. Asano. Lost in Time: A New Temporal Benchmark for VideoLLMs. BMVC 2025. https://arxiv.org/abs/2410.07752
- Categories used (clips): `object_count` (10), `moving_direction` (10), `scene_transition` (10), `action_sequence` (10), `unexpected_action` (10), `action_localization` (10); 60 clips in total.
- Changes: clips re-encoded to H.264/AAC mp4 (long side <= 1280, short side <= 720, audio kept); action_sequence and action_localization cut to the benchmark's permitted `start`..`end` interval; one question per clip; options keep their original order.
- Reproduce: `python -m vwatch.dataset.tvbench --root eval/data --per-category 10 --seed 7`

If you publish results that use T5, cite TVBench:

```bibtex
@inproceedings{cores2025lostintime,
  title     = {Lost in Time: A New Temporal Benchmark for VideoLLMs},
  author    = {Cores, Daniel and Dorkenwald, Michael and Mucientes, Manuel and Snoek, Cees G. M. and Asano, Yuki M.},
  booktitle = {British Machine Vision Conference (BMVC)},
  year      = {2025},
  url       = {https://arxiv.org/abs/2410.07752}
}
```

## T1x: Wikimedia Commons speech videos

Real, openly licensed videos with speech, in six languages. The builder accepts only files whose Commons license is CC BY, CC BY-SA, CC0 or public domain, cuts each to one 35–60 s segment, normalises it to 540 px wide, 30 fps, AAC 44.1 kHz mono, and injects the generic defects (flash frame, frozen frame, jump cut, audio dropout). Modified derivatives keep the license of their source (CC BY-SA derivatives stay CC BY-SA).

| Source id | Commons file | Language | Author | License |
|---|---|---|---|---|
| `steel` | [President Donald J. Trump is Protecting American Steel (short).webm](https://commons.wikimedia.org/wiki/File:President_Donald_J._Trump_is_Protecting_American_Steel_(short).webm) | English | The White House | Public domain |
| `boston` | [President Obama speaks on explosions in Boston (2013-04-15).ogv](https://commons.wikimedia.org/wiki/File:President_Obama_speaks_on_explosions_in_Boston_(2013-04-15).ogv) | English | The White House | Public domain |
| `sem` | [Scanning Electron Microscope (SEM) Lecture 1.webm](https://commons.wikimedia.org/wiki/File:Scanning_Electron_Microscope_(SEM)_Lecture_1.webm) | English | Magnus Hummelgård | CC BY 3.0 |
| `khizr` | [Khizr Khan Trumps TheDonald.webm](https://commons.wikimedia.org/wiki/File:Khizr_Khan_Trumps_TheDonald.webm) | English | Voice of America | Public domain |
| `jharkhand` | [Unheard Voice of India, Jharkhand, 5.ogv](https://commons.wikimedia.org/wiki/File:Unheard_Voice_of_India,_Jharkhand,_5.ogv) | Hindi | Ekta Parishad: Aaquib Nehal Khan, Santosh, Gyan | CC BY-SA 3.0 |
| `gujarat` | [Unheard Voice of India, 7, Gujarat.ogv](https://commons.wikimedia.org/wiki/File:Unheard_Voice_of_India,_7,_Gujarat.ogv) | Hindi | Ekta Parishad: Aaquib Nehal Khan, Santosh, Gyan | CC BY-SA 3.0 |
| `bnreel` | [Can I create a Wikipedia article about myself - Bengali explainer reel.webm](https://commons.wikimedia.org/wiki/File:Can_I_create_a_Wikipedia_article_about_myself_-_Bengali_explainer_reel.webm) | Bengali | Yahya (script, animation, editing; AI-synthesised narration) | CC BY-SA 4.0 |
| `prague` | [Novoroční přání pražského primátora Bohuslava Svobody do roku 2013.ogv](https://commons.wikimedia.org/wiki/File:Novoro%C4%8Dn%C3%AD_p%C5%99%C3%A1n%C3%AD_pra%C5%BEsk%C3%A9ho_prim%C3%A1tora_Bohuslava_Svobody_do_roku_2013.ogv) | Czech | Portál hlavního města Prahy | CC BY 3.0 |
| `linagora` | [Benjamin (linagora).webm](https://commons.wikimedia.org/wiki/File:Benjamin_(linagora).webm) | French | Suzanne Truong WMFr | CC BY 4.0 |
| `pietra` | [Conheça o segredo do sucesso da historiadora, confeiteira e empresária Pietra Diwan.webm](https://commons.wikimedia.org/wiki/File:Conhe%C3%A7a_o_segredo_do_sucesso_da_historiadora,_confeiteira_e_empres%C3%A1ria_Pietra_Diwan.webm) | Portuguese | Chris Delboni | CC BY 3.0 |

Authors and licenses above are as reported by the Commons API when the set was built. The builder writes the authoritative table (with the exact segment used per file) to `eval/data/SOURCES.md`; check it, and the Commons page, before redistributing any derivative.

**Known issue (sets built before v0.2.1):** the builder dropped each source's audio start offset, so speech in the `linagora` videos runs about 0.94 s ahead of the picture and in `prague` about 0.27 s behind it (`pietra` 0.03 s). The defects and their labels sit on each built file's own timeline, so scoring is unaffected; only lip sync is off. Rebuilding with v0.2.1 or later fixes it but changes those videos, so results from the two builds don't mix.

## T1: bring your own videos

T1 uses the maintainer's own Shorts, which are not redistributed. The pilot report names them by source id only (`dosa-reel-DEMO`, `dosa-demo-stick-story`, `dosa-stall-bengaluru-*`).

To build T1 on your own footage, put your clean videos in a folder and point `VWATCH_SHORTS_DIR` at it (default `eval/sources/shorts`), or edit `DEFAULT_SOURCES` in [`src/vwatch/dataset/_source.py`](../src/vwatch/dataset/_source.py) to match your file names. Missing files are skipped with a warning; T1 stops with a clear error if none exist. Each `SourceSpec` takes a name, a path, and (optionally) where burned-in captions sit, so `caption_blocked` knows where to draw; leave the caption fields at their defaults for videos without captions, and only the generic defect types apply. `wrong_number` and the real-video T4 item also need a `props.json` scene list with the on-screen numbers. Then:

```bash
uv run vw dataset build --root eval/data --scale full --seed 7 --tasks T1
```

Use material you have the rights to. If you publish results, describe your sources the same way.

## Procedural clips (T2, T3, T4)

Rendered from scratch with numpy and Pillow (shapes, gradients, text), so there is no third-party footage and the labels are exact. They are covered by the repository's MIT license.
