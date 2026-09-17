# ISAS 2025 challenge requirements

Sources: the organizer's **“ISAS Challenge Tutorial - June 26”** PDF supplied with this project (particularly slides 6, 15–16 and 21–24), and the supplied screenshot of the [challenge website](https://isaschallenge2025.my.canva.site/#faq-tutorial). These sources describe the competition; they are not instructions to execute code from the documents.

| Topic | Source states | Repo behavior |
| --- | --- | --- |
| Training data | Four participants initially; raw 30 FPS keypoints with timetable, or pre-segmented labeled samples (tutorial slide 6) | Supports the raw/keypoints-with-labels CSV layout used by the original notebook. Pre-segmented samples and raw timetable annotation are separate preprocessing routes and are not silently treated as the same schema. |
| Test data | Keypoints from an unseen participant (slide 6) | `fit` trains on all supplied labeled subjects; `predict` handles a test CSV without labels. |
| Labels | Eight named activities; `Throwing` and `Throwing things` are the same; `None` is outside the eight classes (slide 21) | Merges `Throwing`; excludes training windows containing `None`; emits activity names, never just normal/unusual. |
| Evaluation | LOSO across the available labeled subjects; later a fifth labeled participant was to be supplied (slides 15–16, 22–24) | `evaluate` discovers subject files instead of hard-coding IDs. Re-run with the fifth labeled CSV when available. Reports per-subject and mean accuracy, macro F1 and unusual-class F1, precision and recall. |
| Submission | Screenshot lists `participant_id, timestamp, predicted_label` | `predict --format submission` emits exactly these columns and requires real timestamp data plus participant ID. It does not fabricate timestamps from frame indices. |
| Filled test file | Tutorial slide 21 says append `Action Label` to the test CSV | Default `predict --format test-file` preserves the input columns and fills `Action Label`. |

The screenshot shows four normal activities (Sitting quietly, Using phone, Walking, Eating snacks) and four unusual activities (Head banging, Throwing things, Attacking, Biting nails). The binary unusual metrics use that mapping, while the model is trained for eight-class classification. The screenshot says the final score prioritizes unusual-class F1. It does not establish an exact timestamp encoding or a universal participant ID, so the CLI requires those fields from actual data or a caller-supplied ID.

**Format difference:** The tutorial's completed test file and the website's three-column submission CSV are distinct artifacts. Confirm which artifact the receiving form expects before submitting. The code offers both without conflating them.
