Python을 사용하여 동영상의 음성을 텍스트로 변환하고 자동으로 자막 파일(SRT) 을 생성하는 방법을 설명하겠습니다.

1. 주요 라이브러리 설치
아래 라이브러리를 설치해야 합니다.

pip install moviepy pydub openai-whisper faster-whisper srt
moviepy : 동영상에서 오디오를 추출하는 데 사용
pydub : 오디오 파일을 처리하는 데 사용
whisper 또는 faster-whisper : OpenAI의 Whisper 모델을 사용하여 음성을 텍스트로 변환
srt : 자막 파일(SRT)을 생성하는 데 사용
2. 동영상에서 음성 추출 후 텍스트 변환
Python 코드를 실행하여 자동으로 자막을 생성합니다.

편집
import moviepy.editor as mp
import whisper
import srt
from datetime import timedelta

# 1. 동영상에서 음성 추출
def extract_audio(video_path, audio_path="audio.wav"):
    video = mp.VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path)
    return audio_path

# 2. 음성을 텍스트로 변환
def transcribe_audio(audio_path):
    model = whisper.load_model("small")  # whisper 모델 (tiny, small, medium 선택 가능)
    result = model.transcribe(audio_path)
    return result["segments"]  # 변환된 텍스트와 타임스탬프 반환

# 3. SRT 자막 파일 생성
def generate_srt(transcriptions, srt_path="subtitles.srt"):
    subtitles = []
    for i, segment in enumerate(transcriptions):
        start_time = timedelta(seconds=segment["start"])
        end_time = timedelta(seconds=segment["end"])
        text = segment["text"]
        subtitles.append(srt.Subtitle(index=i + 1, start=start_time, end=end_time, content=text))
    
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write(srt.compose(subtitles))
    print(f"SRT 자막이 {srt_path}로 저장되었습니다.")

# 4. 실행 (자동화)
def main(video_path):
    print("Step 1: 음성 추출 중...")
    audio_path = extract_audio(video_path)
    
    print("Step 2: 음성을 텍스트로 변환 중...")
    transcriptions = transcribe_audio(audio_path)
    
    print("Step 3: SRT 자막 생성 중...")
    generate_srt(transcriptions)
    
    print("완료!")

# 실행 예제
if __name__ == "__main__":
    main("example_video.mp4")  # 변환할 동영상 파일 경로
3. 실행 방법
VSCode에서 Python 스크립트 실행

example_video.mp4 파일이 있는 폴더에서 실행
python script.py 명령어 실행
자동으로 진행되는 과정

✅ 동영상에서 음성 추출
✅ Whisper를 사용해 음성을 텍스트로 변환
✅ SRT 자막 파일 생성

생성된 자막 파일 (subtitles.srt) 예제

편집
1
00:00:00,500 --> 00:00:05,000
안녕하세요. 이 강의에서는 파이썬을 사용한 텍스트 변환을 설명합니다.

2
00:00:05,500 --> 00:00:10,000
여러분이 동영상을 텍스트로 변환하는 방법을 배우게 됩니다.
4. 추가 기능
더 빠른 변환
faster-whisper를 사용하면 속도를 개선할 수 있습니다.

편집
from faster_whisper import WhisperModel
model = WhisperModel("small")
segments, _ = model.transcribe(audio_path)
다국어 지원
Whisper는 자동으로 언어를 감지하지만, 명시적으로 설정할 수도 있습니다.

편집
result = model.transcribe(audio_path, language="ko")  # 한국어 변환
음성 인식 품질 향상

"medium" 또는 "large" 모델을 사용하면 인식률이 향상됩니다.
모델 크기에 따라 메모리 사용량이 증가합니다.

💡 요약
✅ Python을 사용해 동영상의 음성을 텍스트로 변환
✅ 자동으로 SRT 자막 파일 생성
✅ Windows + VSCode + Python 환경에서 실행 가능
✅ GPU 없이 CPU 환경에서도 사용 가능

이제 동영상을 넣고 실행하면 자동으로 자막이 생성됩니다! 🚀






