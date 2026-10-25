"""Media helpers for the expanded Week 6 teaching deck."""
from .common import *
from pathlib import Path
import base64

ASSETS=Path(__file__).resolve().parents[2]/'site/slide-assets/week6'

def data_uri(filename,mime):
    return f'data:{mime};base64,'+base64.b64encode((ASSETS/filename).read_bytes()).decode()

def robot_demo_body():
    video=data_uri('aloha-episode0.webm','video/webm')
    poster=data_uri('aloha-frame100.png','image/png')
    b=f'''<foreignObject x="75" y="176" width="800" height="600"><div xmlns="http://www.w3.org/1999/xhtml" class="robot-video" data-deck-ignore-keys="true"><video controls="controls" muted="muted" playsinline="playsinline" preload="metadata" poster="{poster}" aria-label="Recorded ALOHA simulated insertion demonstration, episode zero"><source src="{video}" type="video/webm"/></video><img class="video-print" src="{poster}" alt="ALOHA simulation at two seconds"/></div></foreignObject>'''
    b+=text(945,270,'Two arms. One task.',37,700,color=BLUE)
    b+=lines(945,333,['Coordinate the grippers','to insert the peg.'],32,gap=46)
    b+=line(945,430,1515,430)
    b+=text(945,492,'Watch the feedback loop:',29,700)
    b+=lines(945,545,['approach → align → insert','What must the policy observe?'],29,gap=48)
    b+=text(75,812,'Recorded simulation · episode 0 · original data: 50 Hz; playback sampled at 10 Hz',23,color=MUTED)
    return b
