# -*- coding: utf-8 -*-
"""
将 Plotly 动画 HTML 导出为逐帧 PNG，并合成为 GIF。

修复点：
- 不再单独截图“虚拟初始帧”，而是先 animate 到 'init_circle'（如果存在），保证第一张就是静态帧。
- 若不存在 'init_circle'，则从 frames[0] 开始。
- 使用帧名而非索引播放，避免跳帧。
"""

import os
import time
import base64

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
# 如需手动 chromedriver，可打开下面两行：
# from selenium.webdriver.chrome.service import Service

# ============== 路径配置 ==============
HTML_PATH = "lukai_data.html"
GIF_PATH  = "lukai_data.gif"
FRAME_DIR = "gif_frames"
os.makedirs(FRAME_DIR, exist_ok=True)

# ============== 无头浏览器 ==============
chrome_options = Options()
chrome_options.add_argument("--headless=new")
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--window-size=2400,1200")

# 推荐：Selenium Manager 自动匹配驱动
driver = webdriver.Chrome(options=chrome_options)

# 如需手动 chromedriver：
# service = Service(executable_path="chromedriver.exe")
# driver = webdriver.Chrome(service=service, options=chrome_options)

# ============== 打开 HTML 并等待稳定 ==============
html_full_path = "file:///" + os.path.abspath(HTML_PATH)
driver.get(html_full_path)

# 触发一次 resize，避免响应式布局引入留白
driver.execute_script("window.dispatchEvent(new Event('resize'));")
time.sleep(0.6)

# ============== 获取 Plotly 容器 ID（更稳健） ==============
graph_div_id = driver.execute_script("""
var el = document.querySelector('.js-plotly-plot');
if(!el){
  var cont = document.querySelector('div.plot-container');
  if (cont && cont.parentElement) el = cont.parentElement;
}
if(!el) return null;
if(!el.id){ el.id='gd_'+Math.random().toString(36).slice(2); }
return el.id;
""")
if not graph_div_id:
    driver.quit()
    raise RuntimeError("❌ 无法识别 Plotly 图表容器，请检查 HTML 是否生成正确。")

# 等待 Plotly / frames 可用
for _ in range(20):
    ready = driver.execute_script("""
        var gd=document.getElementById(arguments[0]);
        return !!(window.Plotly && gd && gd._transitionData && gd._transitionData._frames);
    """, graph_div_id)
    if ready:
        break
    time.sleep(0.2)

# 读取帧名（优先使用 name，没有则用索引字符串）
frame_names = driver.execute_script("""
var gd=document.getElementById(arguments[0]);
var fr = (gd && gd._transitionData && gd._transitionData._frames) ? gd._transitionData._frames : [];
return fr.map(function(f,i){ return (f && f.name) ? f.name : String(i); });
""", graph_div_id) or []

if not frame_names:
    driver.quit()
    raise RuntimeError("❌ 没有检测到动画帧(_transitionData._frames)。")

print(f"✅ 动画帧数：{len(frame_names)}")
# 规范化：如果存在 'init_circle'，把它放到序列最前面（避免导出时顺序被改动）
if 'init_circle' in frame_names:
    play_steps = ['init_circle'] + [f for f in frame_names if f != 'init_circle']
else:
    play_steps = frame_names[:]  # 没有 init_circle 就按原顺序

print("▶️ 播放顺序（前5项预览）：", play_steps[:5])

# ============== toImage（异步）脚本 ==============
# 注意：execute_async_script 的最后一个参数是回调
toimage_async_js = r"""
var gd = document.getElementById(arguments[0]);
var cb = arguments[arguments.length - 1];
// 使用图表当前大小
var r = gd.getBoundingClientRect();
var w = Math.round(r.width  || gd.clientWidth  || 1200);
var h = Math.round(r.height || gd.clientHeight || 800);
Plotly.toImage(gd, {format:'png', width:w, height:h, scale:3})
  .then(function(url){ cb(url); })
  .catch(function(err){ cb('ERROR:' + err.toString()); });
"""

screenshot_paths = []

def snapshot_current(idx_for_name: int):
    """截取当前画面；若 toImage 失败则整页截图兜底。"""
    data_url = driver.execute_async_script(toimage_async_js, graph_div_id)
    path = os.path.join(FRAME_DIR, f"frame_{idx_for_name:03d}.png")
    if isinstance(data_url, str) and data_url.startswith("ERROR:"):
        driver.save_screenshot(path)
        print(f"⚠️ 第 {idx_for_name} 帧 toImage 失败，已用整页截图兜底。")
    else:
        header, b64 = data_url.split(',', 1)
        png_bytes = base64.b64decode(b64)
        with open(path, "wb") as f:
            f.write(png_bytes)
    screenshot_paths.append(path)

# ============== 逐帧播放并截图（第一帧强制到 init_circle） ==============
for i, step in enumerate(play_steps):
    # 强制切到目标帧（名字比索引更稳）
    driver.execute_script("""
        var gd=document.getElementById(arguments[0]);
        // 先打断可能的自动播放
        try { Plotly.animate(gd, null, {mode:'immediate'}); } catch(e){}
        Plotly.animate(gd, [arguments[1]], {frame:{duration:0}, transition:{duration:0}, mode:'immediate'});
    """, graph_div_id, step)
    # 等待渲染稳定
    time.sleep(0.55)
    snapshot_current(i)

driver.quit()

# ============== 合成 GIF ==============
from PIL import Image

if not screenshot_paths:
    raise RuntimeError("❌ 没有生成任何帧，无法合成 GIF。")

frames = []
for p in screenshot_paths:
    im = Image.open(p).convert("RGBA")
    # 与白底合成，确保透明像素正确叠色
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
    comp = Image.alpha_composite(bg, im)
    # 转为调色板图，保色 + 抖动
    pal = comp.convert(
        "P",
        palette=Image.ADAPTIVE,
        colors=256,
        dither=Image.FLOYDSTEINBERG,
    )
    frames.append(pal)

# duration: 毫秒/帧（可按你的 HTML 动画节奏调整）
FIRST_FRAME_DURATION = 500   # 第一帧（静态）展示时长
OTHER_FRAME_DURATION = 500   # 其余帧展示时长

durations = [FIRST_FRAME_DURATION] + [OTHER_FRAME_DURATION]*(len(frames)-1)

frames[0].save(
    GIF_PATH,
    save_all=True,
    append_images=frames[1:],
    duration=durations,
    loop=0,
    disposal=2,
    optimize=False
)

print(f"✅ GIF 已生成：{GIF_PATH}")
print(f"🖼️ 共导出帧数：{len(frames)}；首帧：{play_steps[0]!r}")
