from flask import Flask, render_template, request, redirect, url_for, session, g, flash, jsonify
import os, sqlite3, uuid
from datetime import date
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from inference_sdk import InferenceHTTPClient
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont
import time, re

load_dotenv()
from google import genai
from google.genai import types, errors

# Flaskアプリの初期化
app = Flask(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

app.secret_key = os.environ.get("FLASK_SECRET_KEY", "fallback-secret-key-for-dev")
ROBOFLOW_WORKSPACE = "baum-test"
ROBOFLOW_WORKFLOW_ID = "acne-vacne-tvbr3-1-rfdetr-seg-2xlarge-t1-logic"

UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "webp"}
MAX_CONTENT_LENGTH = 8 * 1024 * 1024  # 8MB
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# --- DB接続の安全な管理 ---
DATABASE = "acne_records.db"

def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DATABASE)
        g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def close_db(error):
    db = g.pop("db", None)
    if db is not None:
        db.close()

def init_db():
    with app.app_context():
        db = get_db()
        cur = db.cursor()
        cur.executescript('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT
        );

        CREATE TABLE IF NOT EXISTS acne_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            date TEXT,
            count INTEGER,
            skincare TEXT,
            sleep TEXT,
            diet TEXT,
            image_path TEXT,
            result_image_path TEXT,
            advice TEXT,
            FOREIGN KEY(user_id) REFERENCES users(id)
        );
        ''')
        db.commit()

# Gunicorn起動時でもDBテーブルを初期化
init_db()

def generate_skincare_advice(acne_count, skincare, sleep, diet):
    if not gemini_client:
        return "Gemini APIキーが設定されていません。"

    prompt = f"""
    あなたは親切で専門的な皮膚科のスキンケアアドバイザーです。
    以下のユーザーの今日の記録をもとに、短く丁寧で実践的なスキンケアアドバイス（200文字程度）を作成してください。

    【記録データ】
    - 検出されたニキビ数: {acne_count}個
    - 本日のスキンケア: {skincare or '未入力'}
    - 睡眠時間: {sleep or '未入力'}
    - 食生活: {diet or '未入力'}

    【回答のポイント】
    - 励ましの言葉を含めること
    - 睡眠や食生活、スキンケアに対する具体的な改善アクションを1〜2点提案すること
    """
    max_retries = 3

    for attempt in range(max_retries):
        try:
            # 正しいモデル名を指定
            response = gemini_client.models.generate_content(
                model='models/gemini-3.1-flash-lite',
                contents=prompt,
            )
            return response.text

        except errors.APIError as e:
            if e.code == 429:
                wait_time = 15 if attempt == 0 else 25
                match = re.search(r"Please retry in (\d+(\.\d+)?)s", str(e))
                if match:
                    wait_time = float(match.group(1)) + 1.0

                if attempt < max_retries - 1:
                    app.logger.warning(
                        f"Gemini API Rate Limit (429). {wait_time:.1f}秒後に再試行します... ({attempt + 1}/{max_retries})"
                    )
                    time.sleep(wait_time)
                    continue

            app.logger.error(f"Gemini API Error: {e}")
            break

        except Exception as e:
            app.logger.error(f"Unexpected Error during Gemini call: {e}")
            break

    return "現在AIアドバイスを生成できません。時間を置いて再度お試しください。"

# 実行時に環境変数を動的取得するクライアント生成関数
def get_inference_client():
    api_key = os.environ.get("ROBOFLOW_API_KEY", "")
    return InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key=api_key
    )

def get_predictions_list(result):
    preds = []
    model_size = (640, 640)

    try:
        if isinstance(result, list) and len(result) > 0:
            predictions_outer = result[0].get("predictions", {})
            if isinstance(predictions_outer, dict):
                img_info = predictions_outer.get("image", {})
                w = img_info.get("width", 640)
                h = img_info.get("height", 640)
                model_size = (w, h)

                preds = predictions_outer.get("predictions", [])
    except Exception as e:
        app.logger.error(f"Predictions parsing error: {e}")

    return preds, model_size

def draw_bounding_boxes(image_path, predictions, model_size, output_path):
    image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image.size
    model_w, model_h = model_size

    scale_x = orig_w / model_w if model_w > 0 else 1.0
    scale_y = orig_h / model_h if model_h > 0 else 1.0

    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for pred in predictions:
        if isinstance(pred, dict) and all(k in pred for k in ("x", "y", "width", "height")):
            x = pred["x"] * scale_x
            y = pred["y"] * scale_y
            w = pred["width"] * scale_x
            h = pred["height"] * scale_y

            x1 = x - (w / 2)
            y1 = y - (h / 2)
            x2 = x + (w / 2)
            y2 = y + (h / 2)

            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

            confidence = pred.get("confidence")
            if confidence:
                label = f"{confidence:.2f}"
                draw.text((x1, max(0, y1 - 12)), label, fill="red", font=font)

    image.save(output_path)

# --- 認証 ---
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = generate_password_hash(request.form['password'])

        db = get_db()
        try:
            db.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
            db.commit()
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            return render_template('register.html', error="このユーザー名は既に使用されています。")

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        db = get_db()
        user = db.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()

        if user and check_password_hash(user["password"], password):
            session['user_id'] = user["id"]
            session['username'] = user["username"]
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error="ユーザー名またはパスワードが正しくありません。")

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# --- ルーティング ---
@app.route('/')
def index():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user_id = session['user_id']
    db = get_db()

    def fetch_avg(column):
        query = f"""
            SELECT {column}, AVG(count) 
            FROM acne_records 
            WHERE user_id=? AND {column} IS NOT NULL AND {column} != '' 
            GROUP BY {column}
        """
        rows = db.execute(query, (user_id,)).fetchall()
        return [(row[0], round(row[1], 1)) for row in rows]

    return render_template(
        'index.html',
        username=session['username'],
        skincare_data=fetch_avg('skincare'),
        sleep_data=fetch_avg('sleep'),
        diet_data=fetch_avg('diet')
    )

@app.route('/upload')
def upload():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    if 'file' not in request.files or request.files['file'].filename == '':
        return render_template('upload.html', error="画像ファイルを選択してください。")

    file = request.files['file']
    if not allowed_file(file.filename):
        return render_template('upload.html', error="対応形式: png, jpg, jpeg, gif, webp")

    # 元画像の保存
    original_name = secure_filename(file.filename)
    ext = original_name.rsplit('.', 1)[1].lower()
    file_id = uuid.uuid4().hex
    filename = f"{file_id}.{ext}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    result_filename = f"{file_id}_result.{ext}"
    result_filepath = os.path.join(UPLOAD_FOLDER, result_filename)

    # Roboflow推論
    try:
        inference_client = get_inference_client()
        result = inference_client.run_workflow(
            workspace_name=ROBOFLOW_WORKSPACE,
            workflow_id=ROBOFLOW_WORKFLOW_ID,
            images={"image": filepath},
            use_cache=True
        )

        preds, model_size = get_predictions_list(result)
        acne_count = len(preds)

        draw_bounding_boxes(filepath, preds, model_size, result_filepath)

    except Exception:
        app.logger.exception("Roboflow 推論エラー")
        return render_template('upload.html', error="画像の解析に失敗しました。時間をおいて再試行してください。")

    skincare = request.form.get('skincare', '')
    sleep = request.form.get('sleep', '')
    diet = request.form.get('diet', '')

    # Gemini API でアドバイス生成
    advice = generate_skincare_advice(acne_count, skincare, sleep, diet)

    # DBに保存
    db = get_db()
    cursor = db.cursor()
    cursor.execute(
        """
        INSERT INTO acne_records (user_id, date, count, skincare, sleep, diet, image_path, result_image_path, advice)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (session['user_id'], date.today().isoformat(), acne_count, skincare, sleep, diet, filepath, result_filepath, advice)
    )
    record_id = cursor.lastrowid
    db.commit()

    return redirect(url_for('result_detail', record_id=record_id))

@app.route('/result/<int:record_id>')
def result_detail(record_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))

    db = get_db()
    record = db.execute(
        "SELECT * FROM acne_records WHERE id=? AND user_id=?", 
        (record_id, session['user_id'])
    ).fetchone()

    if not record:
        return redirect(url_for('stats'))

    return render_template('result.html', record=record)

@app.route('/analysis')
def analysis():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user_id = session['user_id']
    db = get_db()

    def fetch_avg(column):
        query = f"""
            SELECT {column}, AVG(count) 
            FROM acne_records 
            WHERE user_id=? AND {column} IS NOT NULL AND {column} != '' 
            GROUP BY {column}
        """
        rows = db.execute(query, (user_id,)).fetchall()
        return [(row[0], round(row[1], 1)) for row in rows]

    return render_template(
        'analysis.html',
        skincare_data=fetch_avg('skincare'),
        sleep_data=fetch_avg('sleep'),
        diet_data=fetch_avg('diet')
    )

@app.route('/stats')
def stats():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    db = get_db()
    records = db.execute(
        "SELECT * FROM acne_records WHERE user_id=? ORDER BY date DESC, id DESC",
        (session['user_id'],)
    ).fetchall()

    graph_records = sorted(records, key=lambda x: x['date'])
    dates = [row["date"] for row in graph_records]
    counts = [row["count"] for row in graph_records]

    return render_template('stats.html', dates=dates, counts=counts, records=records)

@app.route('/calendar')
def calendar():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('calendar.html')

# --- 画像比較ページ ---
@app.route('/compare')
def compare():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user_id = session['user_id']
    db = get_db()
    
    # 選択用のレコード一覧を取得
    records = db.execute(
        "SELECT id, date, count, result_image_path, image_path FROM acne_records WHERE user_id=? ORDER BY date DESC",
        (user_id,)
    ).fetchall()

    # URLパラメータから 2つの ID を取得 (?id1=1&id2=2)
    id1 = request.args.get('id1', type=int)
    id2 = request.args.get('id2', type=int)

    record1 = None
    record2 = None

    if id1 and id2:
        record1 = db.execute("SELECT * FROM acne_records WHERE id=? AND user_id=?", (id1, user_id)).fetchone()
        record2 = db.execute("SELECT * FROM acne_records WHERE id=? AND user_id=?", (id2, user_id)).fetchone()

    return render_template(
        'compare.html',
        records=records,
        record1=record1,
        record2=record2,
        id1=id1,
        id2=id2
    )

@app.route('/api/events')
def api_events():
    if 'user_id' not in session:
        return jsonify([])

    db = get_db()
    records = db.execute(
        "SELECT id, date, count, skincare, sleep, diet FROM acne_records WHERE user_id=?",
        (session['user_id'],)
    ).fetchall()

    events = []
    for r in records:
        count = r["count"]
        bg_color = "#dc3545" if count >= 5 else ("#ffc107" if count >= 2 else "#28a745")

        events.append({
            "id": r["id"],
            "title": f"ニキビ: {count}個",
            "start": r["date"],
            "url": url_for('result_detail', record_id=r["id"]),
            "backgroundColor": bg_color,
            "borderColor": bg_color,
            "extendedProps": {
                "skincare": r["skincare"] or "-",
                "sleep": r["sleep"] or "-",
                "diet": r["diet"] or "-"
            }
        })

    return jsonify(events)

if __name__ == '__main__':
    debug_mode = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(debug=debug_mode)
