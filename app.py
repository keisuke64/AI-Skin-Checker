from flask import Flask, render_template, request, redirect, url_for, session
import os, sqlite3
from datetime import date
from werkzeug.security import generate_password_hash, check_password_hash
from roboflow import Roboflow

app = Flask(__name__)
app.secret_key = "your_secret_key_here"

# --- 初期設定 ---
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Roboflow設定 ---
rf = Roboflow(api_key="bRoUMGutbCpc6SrsErmT")
project = rf.workspace("baum-test").project("skin-trouble-7azan")
model = project.version(2).model


# --- DB接続関数 ---
def get_db():
    conn = sqlite3.connect("acne_records.db")
    conn.row_factory = sqlite3.Row
    return conn

# --- 初期化（テーブル作成） ---
def init_db():
    conn = get_db()
    cur = conn.cursor()
    cur.executescript('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password TEXT
    );
''')
    conn.commit()
    conn.close()
"""
    conn = get_db()
    cur = conn.cursor()
    cur.executescript('''
    CREATE TABLE IF NOT EXISTS acne_records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        date TEXT,
        count INTEGER,
        skincare TEXT,  -- これが追加されていることを確認
        sleep TEXT,     -- これが追加されていることを確認
        diet TEXT,      -- これが追加されていることを確認
        FOREIGN KEY(user_id) REFERENCES users(id)
    );
    ''')
    conn.commit()
    conn.close()
"""

# --- 認証関連 ---
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = generate_password_hash(request.form['password'])

        conn = get_db()
        cur = conn.cursor()
        try:
            cur.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
            conn.commit()
        except sqlite3.IntegrityError:
            conn.close()
            return render_template('register.html', error="このユーザー名は既に使われています。")

        conn.close()
        return redirect(url_for('login'))

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = get_db()
        cur = conn.cursor()
        user = cur.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
        conn.close()

        if user and check_password_hash(user["password"], password):
            session['user_id'] = user["id"]
            session['username'] = user["username"]
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error="ユーザー名またはパスワードが違います。")

    return render_template('login.html')


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


# --- メインページ ---
@app.route('/')
def index():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('base.html', username=session['username'])

# --- アップロードページ ---
@app.route('/upload')
def upload():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    # ファイルの処理
    if 'file' not in request.files or request.files['file'].filename == '':
        return redirect(url_for('upload')) # ファイルがない場合はアップロードページに戻す

    file = request.files['file']
    filename = file.filename
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # Roboflowでニキビ検出
    predictions = model.predict(filepath, confidence=40, overlap=30).json()
    acne_count = len(predictions["predictions"])

    # --- 生活習慣データの取得 ---
    skincare = request.form['skincare']
    sleep = request.form['sleep']
    diet = request.form['diet']
    # --------------------------------

    # DB保存 (skincare, sleep, diet カラムを追加)
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO acne_records (user_id, date, count, skincare, sleep, diet) VALUES (?, ?, ?, ?, ?, ?)",
        (session['user_id'], date.today().isoformat(), acne_count, skincare, sleep, diet)
    )
    conn.commit()
    conn.close()

    # analysis.html は分析ページとして、ここでは結果ページへリダイレクト
    # 新しいデータが記録されたことをメッセージで表示するために upload.html にリダイレクトするのが自然かもしれません
    return redirect(url_for('stats')) # 記録後はグラフ表示ページへ

# --- 集計・分析ページ ---
@app.route('/analysis')
def analysis():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user_id = session['user_id']
    conn = get_db()
    cur = conn.cursor()

    # スキンケア別の平均ニキビ数を集計
    skincare_data_raw = cur.execute(
        "SELECT skincare, AVG(count) FROM acne_records WHERE user_id=? GROUP BY skincare HAVING skincare IS NOT NULL AND skincare != ''",
        (user_id,)
    ).fetchall()
    skincare_data = [(row[0], round(row[1], 1)) for row in skincare_data_raw]

    # 睡眠時間別の平均ニキビ数を集計
    sleep_data_raw = cur.execute(
        "SELECT sleep, AVG(count) FROM acne_records WHERE user_id=? GROUP BY sleep HAVING sleep IS NOT NULL AND sleep != ''",
        (user_id,)
    ).fetchall()
    sleep_data = [(row[0], round(row[1], 1)) for row in sleep_data_raw]

    # 食生活別の平均ニキビ数を集計
    diet_data_raw = cur.execute(
        "SELECT diet, AVG(count) FROM acne_records WHERE user_id=? GROUP BY diet HAVING diet IS NOT NULL AND diet != ''",
        (user_id,)
    ).fetchall()
    diet_data = [(row[0], round(row[1], 1)) for row in diet_data_raw]

    conn.close()

    # index.html または analysis.html にデータを渡してレンダリング
    return render_template(
        'analysis.html',
        skincare_data=skincare_data,
        sleep_data=sleep_data,
        diet_data=diet_data
    )

@app.route('/stats')
def stats():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    conn = get_db()
    cur = conn.cursor()
    records = cur.execute(
        "SELECT date, count FROM acne_records WHERE user_id=? ORDER BY date",
        (session['user_id'],)
    ).fetchall()
    conn.close()

    dates = [row["date"] for row in records]
    counts = [row["count"] for row in records]

    return render_template('stats.html', dates=dates, counts=counts)


if __name__ == '__main__':
    app.run(debug=True)
