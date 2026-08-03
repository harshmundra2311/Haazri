# Haazri AI

Attendance without the roll call. A teacher points a camera at the room (or records ten seconds of the class saying "I am present"), and Haazri figures out who showed up.

The name says it: *haazri* is the register, the daily ritual of calling names and marking who answered. That ritual is trivially gameable — someone shouts "present" for their friend and nobody checks. Haazri replaces the honour system with two biometric signals that are much harder to fake on someone else's behalf: your face and your voice.

Live at **[haazri.streamlit.app](https://haazri.streamlit.app)**.

---

## What it actually does

There are two portals behind one landing page.

**Students** log in with their face. No password, no email, no username — you look at the webcam and you're either recognised or you aren't. If the classifier doesn't know you, the app offers to register you on the spot: type your name, optionally record a voice sample, done. Your face embedding becomes a row in the database and the classifier picks you up from then on.

**Teachers** log in with a normal username and password (bcrypt-hashed), create subjects, and share them via a join code or a QR code. At class time they either:

- **Snap or upload classroom photos** — as many as they want, from a webcam or from disk. Every face in every photo is embedded and matched against the enrolled roster. A student found in *any* photo counts as present, and the results table shows which photo they were found in.
- **Record classroom audio** — everyone says a short phrase, the recording gets split on silence, and each speech segment is matched against the voice profiles of enrolled students.

Either path produces a reviewable table before anything is written. Nothing hits the attendance log until the teacher hits **Confirm & Save**. That review step is deliberate — biometric matching is probabilistic, and a human should get the last word before a student is marked absent.

---

## The recognition pipelines

### Face — [`src/pipelines/face_pipeline.py`](src/pipelines/face_pipeline.py)

Three dlib models do the work, loaded once and cached with `@st.cache_resource` because they are expensive to construct:

| Model | Role |
| --- | --- |
| `dlib.get_frontal_face_detector()` | HOG-based face detection — finds bounding boxes |
| `dlib.shape_predictor` (68-point) | Facial landmark localisation, used to align the face |
| `dlib.face_recognition_model_v1` | ResNet that maps an aligned face to a **128-dimensional embedding** |

Weights come from `face_recognition_models`, installed straight from GitHub (see the requirements note below). The detector runs with an upsample factor of `1`, and `compute_face_descriptor` uses `num_jitters=1`.

The interesting part is the matching. There is an `SVC(kernel='linear', probability=True, class_weight='balanced')` trained over all stored embeddings — but **the SVM is not what decides attendance**. Prediction is pure nearest-neighbour on Euclidean distance in the 128-d space, with two gates:

```python
resemblance_threshold = 0.42
min_margin = 0.08

distances = [np.linalg.norm(x - encoding) for x in X_train]
sorted_idx = np.argsort(distances)
best_idx = int(sorted_idx[0])
best_match_score = distances[best_idx]
predicted_id = int(y_train[best_idx])

is_confident = True
if len(sorted_idx) > 1:
    for idx in sorted_idx[1:]:
        if y_train[idx] != predicted_id:
            runner_up_score = distances[idx]
            is_confident = (runner_up_score - best_match_score) >= min_margin
            break

if best_match_score <= resemblance_threshold and is_confident:
    detected_student[predicted_id] = True
```

Two independent checks have to pass:

1. **Absolute distance** — the nearest embedding must be within `0.42`. The community default for this dlib model is `0.6`; this project tightened it deliberately. At `0.6` the system was generous enough to mark the wrong person present, which is precisely the failure mode the whole app exists to prevent.
2. **Confidence margin** — the nearest match from a *different* student must be at least `0.08` further away. This is the part that catches genuine ambiguity. Two students who look alike will both sit close to a query face; without the margin check, whichever one happened to be marginally closer would win. With it, neither is marked present and the teacher notices the gap.

The bias here is intentional and one-directional: **a false absence is a conversation, a false presence is a broken system.** When the model isn't sure, it stays quiet.

Note the failure semantics of `get_trained_model()` — it returns `None` when no students exist and `0` when students exist but none have embeddings. Both are falsy, so `predict_attendance` short-circuits to an empty result either way rather than crashing on an empty training set. `train_classifier()` clears the resource cache and rebuilds, which is how a newly registered student becomes recognisable without a restart.

### Voice — [`src/pipelines/voice_pipeline.py`](src/pipelines/voice_pipeline.py)

[Resemblyzer](https://github.com/resemble-ai/Resemblyzer)'s `VoiceEncoder` produces a **256-dimensional d-vector** speaker embedding. Audio is loaded through `librosa` at a forced **16 kHz** sample rate — the rate the encoder expects — and passed through `preprocess_wav` for normalisation and VAD trimming.

Single-speaker enrolment is straightforward: embed the utterance, store it. Bulk classroom attendance is where it gets more involved:

```python
segments = librosa.effects.split(audio, top_db=30)

for start, end in segments:
    if (end - start) < sr * 0.5:
        continue
    ...
```

`librosa.effects.split` slices the recording at silences (anything more than 30 dB below peak counts as silence), turning one long recording of a room into per-speaker chunks. Segments shorter than **half a second** are dropped — they're almost always coughs, chair scrapes, or clipped syllables, and embedding them produces noise that can spuriously match someone.

Matching uses **cosine similarity via `np.dot`**, which is valid here specifically because Resemblyzer returns L2-normalised embeddings — the dot product *is* the cosine. The threshold is **0.65**, and when a student is matched by several segments the **highest** score wins:

```python
if sid not in identified_results or score > identified_results[sid]:
    identified_results[sid] = score
```

So a student who speaks three times is judged on their clearest utterance, not dragged down by a mumbled one.

The two pipelines are fully independent — voice attendance works for students who enrolled a voice profile, and enrolled students without one simply can't be marked present that way. The dialog checks for this and refuses to run rather than silently marking a whole class absent.

---

## Architecture

```
Haazri/
├── app.py                              # entry point, routing, deep-link handling
├── requirements.txt
├── .streamlit/
│   └── secrets.toml                    # gitignored — Supabase credentials
└── src/
    ├── database/
    │   ├── config.py                   # Supabase client construction
    │   └── db.py                       # every query in the app lives here
    ├── pipelines/
    │   ├── face_pipeline.py            # dlib detection → 128-d embedding → matching
    │   └── voice_pipeline.py           # Resemblyzer → 256-d d-vector → matching
    ├── screens/
    │   ├── home_screen.py              # student/teacher portal picker
    │   ├── student_screen.py           # FaceID login, registration, dashboard
    │   └── teacher_screen.py           # auth + three-tab dashboard
    ├── components/
    │   ├── dialog_add_photos.py        # camera/upload tabs for classroom photos
    │   ├── dialog_attendance_results.py# review-before-save table
    │   ├── dialog_auto_enroll.py       # QR/deep-link enrolment confirmation
    │   ├── dialog_create_subject.py
    │   ├── dialog_enroll.py            # manual join-code enrolment
    │   ├── dialog_share_subject.py     # QR generation
    │   ├── dialog_voice_attendance.py  # record → segment → match → review
    │   ├── header.py / footer.py
    │   └── subject_card.py
    └── ui/
        └── base_layout.py              # injected CSS
```

There is no ORM, no service layer, no dependency injection. Every Supabase call is a plain function in [`src/database/db.py`](src/database/db.py), pipelines are stateless module-level functions, and screens compose components. For a project this size that flatness is a feature — you can trace any behaviour from button to query in about two hops.

### Routing

Streamlit has no router, so [`app.py`](app.py) uses a `match` on `st.session_state['login_type']`:

```python
match st.session_state['login_type']:
    case 'teacher':  teacher_screen()
    case 'student':  student_screen()
    case None:       home_screen()
```

Session state also carries `is_logged_in`, `user_role`, `teacher_data` / `student_data`, `current_teacher_tab`, and `attendance_images`. Logout deletes the data key and flips the flag.

### QR deep links

The share dialog builds `https://proxzero.streamlit.app/?join-code=CS101` and renders it as a QR with **segno** (pure-Python, no system dependencies) at scale 10, border 1, straight into an in-memory `BytesIO` buffer — nothing touches disk.

The receiving end sits at the bottom of `app.py` and handles the awkward case where someone scans a class QR while logged out or logged in as a teacher:

```python
join_code = st.query_params.get('join-code')
if join_code:
    if st.session_state.login_type != 'student':
        st.session_state.login_type = 'student'
        st.rerun()
    if st.session_state.get('is_logged_in') and st.session_state.get('user_role') == 'student':
        auto_enroll_dialog(join_code)
```

It forces the student portal, waits for FaceID login to complete, *then* fires the enrolment dialog — the join code survives the login round-trip because it lives in the URL, not in session state. Every exit path from that dialog calls `st.query_params.clear()` so the code doesn't re-fire the dialog on the next rerun.

---

## Data model (Supabase / PostgreSQL)

| Table | Columns | Notes |
| --- | --- | --- |
| `teachers` | `teacher_id`, `username`, `password`, `name` | `password` is a bcrypt hash; `username` uniqueness enforced in app code |
| `students` | `student_id`, `name`, `face_embedding`, `voice_embedding` | embeddings stored as JSON float arrays (128-d and 256-d); no password |
| `subjects` | `subject_id`, `subject_code`, `name`, `section`, `teacher_id` | `subject_code` doubles as the human-typed join code |
| `subject_students` | `student_id`, `subject_id` | join table for enrolment |
| `attendance_logs` | `student_id`, `subject_id`, `timestamp`, `is_present` | one row per student per session — absences are recorded explicitly |

A few deliberate choices worth calling out:

**Absences are rows, not missing rows.** Every session writes a record for *every* enrolled student with `is_present` true or false. This makes "attended 7 of 12 classes" a straightforward count instead of requiring a join against a separate sessions table.

**`timestamp` is the session key.** All logs from one attendance run share an identical `datetime.now().strftime("%Y-%m-%dT%H:%M:%S")` string, so counting distinct timestamps counts distinct classes:

```python
unique_sessions = len(set(log['timestamp'] for log in attendance))
```

The seconds-level truncation is load-bearing — microsecond precision would make every row its own "session."

**Embedded aggregation over N+1 queries.** `get_teacher_subjects` leans on PostgREST's embedding syntax to pull subjects, enrolment counts, and attendance timestamps in a single round trip:

```python
supabase.table('subjects').select("*, subject_students(count), attendance_logs(timestamp)")
```

`get_attendance_for_teacher` uses `subjects!inner(*)` — an **inner** join, so it filters logs by the teacher who owns the subject rather than fetching everything and filtering in Python.

Students authenticate by face alone: there is no password column on `students` by design. The face *is* the credential.

---

## Tech stack

- **Python 3.12** — the deployment target. The codebase uses `match` statements (3.10+) and nested same-quote f-strings like `f'{sub['name']}'` (3.12+), so 3.12 is a hard floor, not a preference
- **Streamlit** — UI, session state, camera and microphone input, and the dialog system (`@st.dialog`)
- **Supabase** (PostgreSQL + PostgREST) — database and API
- **dlib** (`dlib-bin`) + `face_recognition_models` — face detection, landmarks, embeddings
- **Resemblyzer** + **librosa** — speaker embeddings and audio segmentation
- **scikit-learn** — `SVC` classifier
- **NumPy / pandas** — vector math and results tables
- **bcrypt** — teacher password hashing
- **segno** — QR generation
- **Pillow** — image decoding

### On `requirements.txt`

```
setuptools<70.0.0
git+https://github.com/ageitgey/face_recognition_models
dlib-bin
numba>=0.58
llvmlite>=0.41
```

Every one of those pins exists for a reason, and the last deployment fix was largely about them:

- **`dlib-bin` instead of `dlib`** — prebuilt wheels. Plain `dlib` compiles from source and needs CMake plus a full C++ toolchain, which is a slow and fragile way to fail a cloud build.
- **`setuptools<70.0.0`** — `face_recognition_models` ships a legacy `setup.py` that setuptools 70 refuses to install.
- **`git+https://...face_recognition_models`** — the model weights aren't on PyPI, so they're pulled from source. It also keeps ~100 MB of binaries out of this repo.
- **`numba>=0.58` / `llvmlite>=0.41`** — Resemblyzer transitively pins `numba==0.53.1`, which has no wheels for Python 3.12 and fails to build from source on Streamlit Cloud. These floors override that pin and drag `llvmlite` (numba's JIT backend) up with it. This was the dependency conflict fixed in commit `9c5fa6f`.

---

## Running it locally

```bash
git clone <repo-url>
cd Haazri

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
```

Create `.streamlit/secrets.toml` (gitignored — never commit it):

```toml
SUPABASE_URL = "https://your-project.supabase.co"
SUPABASE_KEY = "your-anon-key"
```

Then:

```bash
streamlit run app.py
```

Your browser needs to grant **camera and microphone** permissions. Streamlit's `camera_input` and `audio_input` require a secure context — `localhost` counts, so local development works without HTTPS.

Getting to a working demo: register a teacher account → create a subject → open the student portal and register a face (add a voice sample too, if you want to try voice attendance) → enrol via the subject code or QR → back on the teacher side, take attendance.

### Deployment

Deployed to **Streamlit Community Cloud** at `proxzero.streamlit.app`. Secrets are set through the app dashboard rather than a file. The domain is hardcoded in [`src/components/dialog_share_subject.py`](src/components/dialog_share_subject.py) as `app_domain`, so a different host means changing it there for QR links to resolve.

---

## Design notes and known rough edges

Things worth knowing before you extend this:

**Camera reruns.** Streamlit reruns the entire script on every interaction, and `camera_input` holds its last frame — so a naive implementation re-runs face recognition on every rerun forever. Two guards handle this: an identity check against the previously processed photo, and an incrementing `login_cam_key` that resets the widget.

```python
if photo_source and photo_source is not st.session_state.get('login_last_photo'):
    st.session_state.login_last_photo = photo_source
```

**Model caching.** `@st.cache_resource` on the dlib loaders and the trained classifier is what makes the app usable — without it, every rerun reloads the ResNet. The tradeoff is that new registrations need an explicit `st.cache_resource.clear()` via `train_classifier()`.

**The SVM is currently vestigial.** `get_trained_model()` fits an `SVC` but `predict_attendance` only consumes `X` and `y` for its distance computation. The classifier is trained and never called. It's kept because thresholded nearest-neighbour is far easier to reason about and tune than SVM probability outputs — and with one embedding per student, an SVM has almost nothing to learn anyway. If per-student sample counts grow, it's already wired in.

**One embedding per student.** Registration stores a single face embedding from a single frame. Multiple embeddings per student under varied lighting and angles would meaningfully improve recall, and the schema would need to change to support it.

**Thresholds are global constants.** `0.42` / `0.08` / `0.65` are hardcoded. They were tuned by hand and are the right knobs to turn first if accuracy is off — raise the face threshold for more recall, lower it for more precision.

**Uniqueness is enforced in app code, not the schema.** Teacher usernames are checked with a `SELECT` before insert, and subject codes aren't checked at all. Under concurrency, database-level unique constraints would be the correct fix.

**Subject cards render one card.** In `teacher_tab_manage_subjects`, the loop building `stats` closes before `subject_card` is called, so only the last subject renders on the manage tab.

**Face embeddings are biometric data.** They live in Supabase as plain JSON arrays. Anything beyond a classroom demo should mean row-level security, encryption at rest, explicit consent, and a deletion path.

---

## Roadmap

- Multiple face embeddings per student, captured across angles and lighting
- Liveness detection — a printed photo currently passes
- Fusing face and voice signals into one confidence score instead of two separate flows
- Per-subject threshold configuration
- CSV/PDF export for attendance records
- Database-level unique constraints and row-level security

---

Built by **Harsh Mundra**.