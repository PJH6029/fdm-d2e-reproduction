# G002 FDM-1/D2E data universe and split contract

**Canonical roadmap:** `ROADMAP.md`.

## Dataset pin

- Primary dataset: `open-world-agents/D2E-480p`
- Pinned revision: `f075f7e25df6f6d385840a836f86bf92dfb877ff`
- License: `cc-by-nc-4.0`
- Games: `29`
- 480p recording pairs: `459`
- Published D2E-480p README summary hours: `268.7`
- Published D2E-480p per-game table hour sum: `269.8`
- Dataset fingerprint: `a4d96c1b35a770a5ce426a3407af4fbd728ddac9a648cdb4746e2f827f803216`

## Split artifacts

| Split | Manifest | Counts |
| --- | --- | --- |
| Recording-level in-distribution 80/10/10 | `artifacts/sources/fdm1_d2e_recording_level_split_manifest.json` | `{"test": 45, "train": 368, "val": 46}` |
| Held-out game category coverage | `artifacts/sources/fdm1_d2e_heldout_game_split_manifest.json` | `{"heldout_game_test": 47, "train_pool": 412}` |
| Pseudo-label simulation | `artifacts/sources/fdm1_d2e_pseudo_label_split_manifest.json` | `{"D_FDM_GT_EVAL": 74, "D_IDM_LABELED_A": 187, "D_PSEUDO_B": 107}` |
| Data scale | `artifacts/sources/fdm1_d2e_scale_split_manifest.json` | `1pct, 5pct, 10pct, 25pct, 50pct, 100pct` |

## Held-out game roles

| Role | Game |
| --- | --- |
| `fps_or_shooter` | `Battlefield_6_Open_Beta` |
| `sandbox_or_open_world` | `Minecraft_1.21.8` |
| `top_down_or_2d` | `Stardew_Valley` |
| `ui_or_slower_paced` | `MapleStory_Worlds_Southperry` |
| `low_resource` | `PEAK` |

## Per-game metadata and coarse action statistics

| Game | Category | Tags | README hours | 480p recs | Decoded events | 50ms windows | Events/hour |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Apex_Legends | fps_shooter | fps, shooter, battle_royale, high_mouse_motion | 25.6 | 36 | 8818585 | 1820649 | 348742.735 |
| Barony | first_person_roguelike | first_person, dungeon_crawler, roguelike, survival | 9.3 | 11 | 12031742 | 665733 | 1301250.49 |
| Battlefield_6_Open_Beta | fps_shooter | fps, shooter, vehicle, large_map | 2.2 | 7 | 3546415 | 158077 | 1615300.563 |
| Brotato | top_down_2d_action | top_down, 2d_action, roguelike, high_event_density | 6.0 | 13 | 3822947 | 427049 | 644544.746 |
| Core_Keeper | top_down_sandbox | top_down, sandbox, crafting, survival | 8.9 | 19 | 3480087 | 638594 | 392371.784 |
| Counter-Strike_2 | fps_shooter | fps, tactical_shooter, high_mouse_motion | 9.9 | 10 | 6994283 | 709660 | 709619.212 |
| Cyberpunk_2077 | open_world | first_person, open_world, rpg, driving_segments | 14.2 | 33 | 9997095 | 1006073 | 715445.943 |
| Dinkum | life_sim | life_sim, farming, sandbox, slower_paced | 10.4 | 33 | 8692258 | 734693 | 851842.339 |
| Eternal_Return | top_down_2d_action | top_down, moba, battle_royale, ui_heavy | 17.1 | 30 | 13700266 | 1217283 | 810344.983 |
| Euro_Truck_Simulator_2 | driving_vehicle | driving, vehicle, simulation, low_frequency_controls | 19.6 | 35 | 6871873 | 1402283 | 352835.234 |
| Grand_Theft_Auto_V | open_world | third_person, open_world, driving, shooter | 12.9 | 11 | 7804314 | 810024 | 693696.267 |
| Grounded | sandbox_survival | first_person, sandbox, survival, crafting | 9.7 | 31 | 10232608 | 673143 | 1094489.287 |
| MapleStory_Worlds_Southperry | side_scroller_ui | side_scroller, 2d, mmo, ui_heavy, keyboard_heavy | 14.1 | 8 | 3511267 | 1011269 | 249994.037 |
| Medieval_Dynasty | open_world_survival | first_person, open_world, survival, life_sim | 10.9 | 12 | 8579281 | 781730 | 790181.058 |
| Minecraft_1.21.8 | sandbox_survival | first_person, sandbox, crafting, survival | 8.6 | 10 | 10683627 | 619416 | 1241849.006 |
| Monster_Hunter_Wilds | third_person_action | third_person, action_rpg, open_area, controller_like_keyboard_mouse | 7.9 | 34 | 6551064 | 561104 | 840622.442 |
| OguForest | top_down_2d_action | top_down, 2d, adventure, low_resource | 0.8 | 1 | 212435 | 60629 | 252277.426 |
| PEAK | first_person_traversal | first_person, traversal, survival, low_resource | 1.8 | 9 | 2257945 | 121812 | 1334614.587 |
| PUBG | fps_shooter | fps, shooter, battle_royale, high_mouse_motion | 4.9 | 13 | 3219577 | 346408 | 669180.72 |
| Raft | sandbox_survival | first_person, sandbox, survival, crafting | 10.8 | 21 | 15280936 | 734385 | 1498161.531 |
| Rainbow_Six | fps_shooter | fps, tactical_shooter, ui_heavy, high_mouse_motion | 13.7 | 11 | 5922900 | 986219 | 432407.816 |
| Ready_Or_Not | fps_shooter | fps, tactical_shooter, slow_tactical, high_mouse_motion | 9.6 | 16 | 7937661 | 686147 | 832928.831 |
| Satisfactory | sandbox_factory | first_person, sandbox, factory, crafting, ui_heavy | 9.8 | 22 | 8248839 | 693866 | 855952.582 |
| Skul | side_scroller_2d_action | side_scroller, 2d_action, platformer_like, low_resource | 2.0 | 1 | 475002 | 141386 | 241892.067 |
| Slime_Rancher | open_world_sandbox | first_person, open_world, sandbox, slower_paced | 10.7 | 10 | 12236653 | 767202 | 1148379.493 |
| Stardew_Valley | life_sim | top_down, farming, life_sim, ui_heavy, slower_paced | 14.6 | 13 | 5563315 | 1038965 | 385536.272 |
| Super_Bunny_Man | side_scroller_2d_action | side_scroller, 2d, platformer_like, physics, low_resource | 0.7 | 3 | 172592 | 49334 | 251887.787 |
| VALORANT | fps_shooter | fps, tactical_shooter, high_mouse_motion, low_resource | 0.3 | 1 | 235989 | 18063 | 940663.677 |
| Vampire_Survivors | top_down_2d_action | top_down, 2d_action, roguelike, low_resource | 2.8 | 5 | 779793 | 199460 | 281485.468 |

## Validation

- Status: `pass`
- Error count: `0`
- Findings: `[]`

## Claim boundary

G002 proves only dataset pinning, all-game inventory, coarse pre-tokenization action/window statistics, and leakage-safe split manifests. It does not prove decoding correctness, action-tokenization correctness, model training, baseline wins, harness stability, or FDM-1 parity.
