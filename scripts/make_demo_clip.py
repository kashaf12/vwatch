"""Rebuild the README demo clip: 42 s of William Fisher, CopyrightX Lecture 3.2 (BerkmanCenter, CC BY 3.0,
Wikimedia Commons) with three planted glitches. Usage: make_demo_clip.py SOURCE.webm STRAY_FRAME.png OUT.mp4
(the stray frame is the lecture at 2:30). Times below are in output time and form the answer key."""
import subprocess, sys
src, stray, out = sys.argv[1:4]
SS, DUR = 15.0, 42.2
FL, DO, DOLEN, FZ, HOLD = 6.00, 13.00, 0.6, 33.00, 1.2
n0 = round(FL * 30)
fc = (f"[0:v]fps=30,scale=1920:1080,setsar=1[v0];[v0][1:v]overlay=0:0:enable='between(n,{n0},{n0+1})'[v1];"
      f"[v1]split[s1][s2];[s1]trim=0:{FZ},setpts=PTS-STARTPTS,tpad=stop_mode=clone:stop_duration={HOLD}[p];"
      f"[s2]trim={FZ+HOLD},setpts=PTS-STARTPTS[q];[p][q]concat=n=2:v=1[v];"
      f"[0:a]aresample=48000,volume=enable='between(t,{DO},{DO+DOLEN})':volume=0[a]")
subprocess.run(["ffmpeg","-v","error","-y","-ss",str(SS),"-t",str(DUR),"-i",src,"-i",stray,"-filter_complex",fc,
                "-map","[v]","-map","[a]","-c:v","libx264","-preset","medium","-crf","20","-pix_fmt","yuv420p",
                "-c:a","aac","-b:a","160k","-movflags","+faststart",out],check=True)
print(f"flash {n0/30:.2f}-{(n0+2)/30:.2f}  dropout {DO:.2f}-{DO+DOLEN:.2f}  freeze {FZ:.2f}-{FZ+HOLD:.2f}")
