import os
import json
import sys
import time
import base64
import threading
import queue
import logging
import urllib.parse
import requests
from io import BytesIO
from PIL import Image
from google import genai
from google.genai import types

# ============ الإعدادات (Config) ============
GOOGLE_MODELS = ["gemini-3.1-flash-image", "gemini-3.1-flash-lite-image"]
GOOGLE_RPM_PER_KEY = int(os.getenv("GOOGLE_RPM_PER_KEY", "8"))
BACKUP_RPM = int(os.getenv("BACKUP_RPM", "15"))
GOOGLE_INTERVAL = 60.0 / GOOGLE_RPM_PER_KEY
BACKUP_INTERVAL = 60.0 / BACKUP_RPM
MAX_ATTEMPTS_PER_TASK = 6
MAX_BACKUP_RETRIES = 3
GOOGLE_REQUEST_TIMEOUT_MS = 30000
STALL_TIMEOUT_SECONDS = 600
KEY_STATE_FILE = "key_state.json"
THREAD_JOIN_TIMEOUT = 20  # ثانية لكل ثريد عند الإغلاق

# ============ إعداد السجلات ============
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("production.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger("image_factory")


class TemporaryFailure(Exception):
    pass


class PermanentFailure(Exception):
    pass


# ============ حالة المفاتيح المشتركة ============
state_lock = threading.Lock()
file_lock = threading.Lock()
key_status = {}
key_fail_streak = {}


def load_dead_keys():
    if not os.path.exists(KEY_STATE_FILE):
        return set()
    try:
        with open(KEY_STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("date") == time.strftime("%Y-%m-%d"):
            return set(data.get("dead_keys", []))
    except Exception:
        pass
    return set()


def save_dead_keys():
    with state_lock:
        dead = [k for k, v in key_status.items() if v == "dead"]
    payload = {"date": time.strftime("%Y-%m-%d"), "dead_keys": dead}
    tmp_path = KEY_STATE_FILE + ".tmp"
    with file_lock:
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, KEY_STATE_FILE)
        except Exception as e:
            log.warning(f"تعذر حفظ حالة المفاتيح: {e}")


def mark_key_dead(key_name, reason):
    with state_lock:
        key_status[key_name] = "dead"
    log.error(f"[{key_name}] تم تعطيل المفتاح نهائيًا | السبب: {reason}")
    save_dead_keys()


def alive_keys_count():
    with state_lock:
        return sum(1 for v in key_status.values() if v == "alive")


def register_failure_and_get_backoff(key_name):
    with state_lock:
        streak = key_fail_streak.get(key_name, 0) + 1
        key_fail_streak[key_name] = streak
    return min(1 * (2 ** (streak - 1)), 30)


def reset_backoff(key_name):
    with state_lock:
        key_fail_streak[key_name] = 0


# ============ الطوابير ============
main_queue = queue.Queue()
backup_queue = queue.Queue()
failed_tasks = []
failed_lock = threading.Lock()
last_progress_time = time.time()
progress_lock = threading.Lock()

in_flight_lock = threading.Lock()
in_flight_count = 0


def in_flight_inc():
    global in_flight_count
    with in_flight_lock:
        in_flight_count += 1


def in_flight_dec():
    global in_flight_count
    with in_flight_lock:
        in_flight_count -= 1


def get_in_flight():
    with in_flight_lock:
        return in_flight_count


def touch_progress():
    global last_progress_time
    with progress_lock:
        last_progress_time = time.time()


def seconds_since_progress():
    with progress_lock:
        return time.time() - last_progress_time


def init_google_clients():
    keys = []
    env_names = ["GEMINI_API_KEY", "GEMINI_API_KEY_2", "GEMINI_API_KEY_3", "GEMINI_API_KEY_4"]
    for name in env_names:
        key = os.getenv(name)
        if key and key.strip():
            keys.append((name, key.strip()))
    return keys


def try_google_generate(client, prompt):
    last_err = None
    for model_name in GOOGLE_MODELS:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE"],
                    image_config=types.ImageConfig(aspect_ratio="16:9")
                )
            )
            if response.candidates:
                for candidate in response.candidates:
                    if candidate.content and candidate.content.parts:
                        for part in candidate.content.parts:
                            if part.inline_data and part.inline_data.data:
                                data = part.inline_data.data
                                return base64.b64decode(data) if isinstance(data, str) else data
        except Exception as e:
            last_err = e
            err_str = str(e)
            err_lower = err_str.lower()

            if any(x in err_str for x in ["API_KEY_INVALID", "PERMISSION_DENIED", "UNAUTHENTICATED"]):
                raise PermanentFailure(err_str)

            if "limit: 0" in err_str:
                raise PermanentFailure(f"Quota is 0 for this model: {err_str}")

            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                if "retry in" in err_lower or "retrydelay" in err_lower:
                    raise TemporaryFailure(err_str)
                if any(x in err_lower for x in ["daily", "per day"]):
                    raise PermanentFailure(f"استُنفدت الحصة اليومية: {err_str}")
                raise TemporaryFailure(err_str)

            continue

    raise TemporaryFailure(str(last_err) if last_err else "لم يرجع الرد أي صورة")


def try_pollinations_generate(prompt):
    encoded = urllib.parse.quote(prompt)
    seed = int(time.time() * 1000) % 999999
    url = f"https://image.pollinations.ai/prompt/{encoded}?width=1280&height=720&model=flux&nologo=true&seed={seed}"
    response = requests.get(url, timeout=50)
    if response.status_code == 200:
        return response.content
    raise TemporaryFailure(f"HTTP {response.status_code}")


def save_processed_image(image_bytes, output_path):
    if not image_bytes or len(image_bytes) < 100:
        raise TemporaryFailure("بيانات الصورة فارغة أو تالفة")
    try:
        image = Image.open(BytesIO(image_bytes))
        image.load()
    except Exception as e:
        raise TemporaryFailure(f"فشل فك تشفير الصورة: {e}")

    image = image.convert("RGB")
    image = image.resize((1280, 720), Image.Resampling.LANCZOS)

    tmp_output = output_path + ".part"
    image.save(tmp_output, "PNG", quality=95)
    os.replace(tmp_output, output_path)


def google_worker(key_name, key_val):
    client = genai.Client(
        api_key=key_val,
        http_options=types.HttpOptions(timeout=GOOGLE_REQUEST_TIMEOUT_MS)
    )
    with state_lock:
        key_status[key_name] = "alive"

    while True:
        try:
            task = main_queue.get(timeout=2)
        except queue.Empty:
            if alive_keys_count() == 0:
                return
            continue

        if task is None:
            main_queue.put(None)
            return

        task.setdefault("tried_keys", set())

        if key_name in task["tried_keys"]:
            if len(task["tried_keys"]) >= alive_keys_count():
                backup_queue.put(task)
            else:
                main_queue.put(task)
                time.sleep(0.3)
            continue

        in_flight_inc()
        try:
            img_bytes = try_google_generate(client, task["prompt"])
            save_processed_image(img_bytes, task["output_path"])
            log.info(f"[{task['id']}] نجاح -> Google ({key_name})")
            reset_backoff(key_name)
            touch_progress()
            time.sleep(GOOGLE_INTERVAL)

        except PermanentFailure as e:
            mark_key_dead(key_name, str(e))
            main_queue.put(task)
            if alive_keys_count() == 0:
                while True:
                    try:
                        leftover = main_queue.get_nowait()
                    except queue.Empty:
                        break
                    if leftover is not None:
                        backup_queue.put(leftover)
            return

        except TemporaryFailure as e:
            task["tried_keys"].add(key_name)
            task["attempts"] = task.get("attempts", 0) + 1
            log.warning(f"[{task['id']}] فشل مؤقت مع {key_name}: {e}")

            backoff = register_failure_and_get_backoff(key_name)

            if len(task["tried_keys"]) >= alive_keys_count() or task["attempts"] >= MAX_ATTEMPTS_PER_TASK:
                log.warning(f"[{task['id']}] استنفاد مفاتيح جوجل -> تحويل للاحتياطي")
                backup_queue.put(task)
            else:
                main_queue.put(task)

            time.sleep(backoff)

        finally:
            in_flight_dec()


def backup_worker():
    last_sent_at = 0.0
    while True:
        task = backup_queue.get()
        if task is None:
            return

        elapsed = time.time() - last_sent_at
        if elapsed < BACKUP_INTERVAL:
            time.sleep(BACKUP_INTERVAL - elapsed)

        in_flight_inc()
        try:
            img_bytes = try_pollinations_generate(task["prompt"])
            save_processed_image(img_bytes, task["output_path"])
            log.info(f"[{task['id']}] نجاح -> الاحتياطي (Pollinations)")
            touch_progress()
        except Exception as e:
            task["backup_attempts"] = task.get("backup_attempts", 0) + 1
            log.warning(f"[{task['id']}] فشل الاحتياطي: {e}")
            if task["backup_attempts"] >= MAX_BACKUP_RETRIES:
                with failed_lock:
                    failed_tasks.append(task["id"])
                log.error(f"[{task['id']}] فشل نهائي بعد استنفاد كل المحاولات")
                touch_progress()
            else:
                backup_queue.put(task)
        finally:
            last_sent_at = time.time()
            in_flight_dec()


def build_tasks(episode):
    tasks = []
    for item in episode.get("timeline", []):
        out_path = f"images/timeline/{item['id']:03d}.png"
        if os.path.exists(out_path):
            continue
        tasks.append({"id": item["id"], "prompt": item["image_prompt"], "output_path": out_path})

    for idx, thumb in enumerate(episode.get("thumbnails", [])):
        angle = thumb.get("angle", f"angle_{idx+1}")
        out_path = f"images/thumbnails/thumb_{angle}.png"
        if os.path.exists(out_path):
            continue
        tasks.append({"id": f"thumb_{angle}", "prompt": thumb.get("prompt", ""), "output_path": out_path})

    return tasks


def dump_failed_tasks():
    with failed_lock:
        if failed_tasks:
            with open("failed_tasks.json", "w", encoding="utf-8") as f:
                json.dump(failed_tasks, f, ensure_ascii=False, indent=2)
            log.error(f"تم حفظ {len(failed_tasks)} مهمة فاشلة في failed_tasks.json")


def main():
    if not os.path.exists("current_episode.json"):
        log.error("current_episode.json غير موجود")
        sys.exit(1)

    with open("current_episode.json", "r", encoding="utf-8") as f:
        episode = json.load(f)

    google_keys = init_google_clients()
    if not google_keys:
        log.error("لا يوجد أي مفتاح Google متاح")
        sys.exit(1)

    dead_from_before = load_dead_keys()
    if dead_from_before:
        log.info(f"مفاتيح معطوبة من تشغيل سابق اليوم: {dead_from_before}")

    os.makedirs("images/timeline", exist_ok=True)
    os.makedirs("images/thumbnails", exist_ok=True)

    tasks = build_tasks(episode)
    if not tasks:
        log.info("لا توجد صور جديدة للإنتاج (جميع الملفات موجودة بالفعل)")
        return

    active_google_keys = [(n, v) for n, v in google_keys if n not in dead_from_before]
    for n, _ in google_keys:
        if n in dead_from_before:
            with state_lock:
                key_status[n] = "dead"

    if not active_google_keys:
        log.warning("كل مفاتيح Google معطوبة من قبل -> التحويل المباشر للاحتياطي")
        for t in tasks:
            backup_queue.put(t)
    else:
        for task in tasks:
            main_queue.put(task)

    log.info(
        f"بدء الإنتاج: {len(tasks)} صورة | مفاتيح Google الحية: {len(active_google_keys)} "
        f"| حد كل مفتاح: {GOOGLE_RPM_PER_KEY}/دقيقة | حد الاحتياطي: {BACKUP_RPM}/دقيقة"
    )

    threads = []
    for key_name, key_val in active_google_keys:
        t = threading.Thread(target=google_worker, args=(key_name, key_val), daemon=True)
        t.start()
        threads.append(t)

    backup_thread = threading.Thread(target=backup_worker, daemon=True)
    backup_thread.start()

    total_tasks = len(tasks)
    start_time = time.time()
    touch_progress()

    try:
        while True:
            done_count = sum(1 for t in tasks if os.path.exists(t["output_path"]))
            with failed_lock:
                finished_count = done_count + len(failed_tasks)

            if finished_count >= total_tasks:
                break

            if (alive_keys_count() == 0 and main_queue.empty()
                    and backup_queue.empty() and get_in_flight() == 0):
                break

            if seconds_since_progress() > STALL_TIMEOUT_SECONDS:
                log.error(f"توقف كامل لأكثر من {STALL_TIMEOUT_SECONDS} ثانية بدون أي تقدم -> إيقاف اضطراري")
                break

            time.sleep(3)
    except KeyboardInterrupt:
        log.warning("تم إيقاف السكربت يدويًا")

    # إرسال إشارة التوقف لجميع ثريدز جوجل وعامل الاحتياطي
    for _ in range(len(active_google_keys)):
        main_queue.put(None)
    backup_queue.put(None)

    for t in threads:
        t.join(timeout=THREAD_JOIN_TIMEOUT)
        if t.is_alive():
            log.warning(f"ثريد لم يتوقف خلال {THREAD_JOIN_TIMEOUT}s")
            
    backup_thread.join(timeout=THREAD_JOIN_TIMEOUT)
    if backup_thread.is_alive():
        log.warning(f"ثريد الاحتياطي لم يتوقف خلال {THREAD_JOIN_TIMEOUT}s")

    elapsed = round(time.time() - start_time, 2)
    done_count = sum(1 for t in tasks if os.path.exists(t["output_path"]))
    dump_failed_tasks()

    log.info("=======================================================")
    log.info(f"انتهى التشغيل في {elapsed} ثانية | نجاح: {done_count}/{total_tasks} | فشل: {total_tasks - done_count}")
    log.info("=======================================================")

    if done_count < total_tasks:
        sys.exit(1)


if __name__ == "__main__":
    main()
