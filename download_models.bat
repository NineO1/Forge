cd /d I:\APP-Genesis\Forge
call venv\Scripts\activate.bat

echo [1/4] Caching SDXL...
python -c "from forge.generate import text_to_image; text_to_image('a purple cat DJ', r'.', filename='_cache_test.png')"
del _cache_test.png 2>nul

echo [2/4] Caching LTX-Video...
python -c "from forge.generate import text_to_video; text_to_video('a purple cat dancing', r'.', filename='_cache_test.mp4', num_frames=9)"
del _cache_test.mp4 2>nul

echo [3/4] Caching Whisper...
python -c "from forge.transcribe import transcribe_to_srt; transcribe_to_srt(r'I:\Nvidia\videos\test.mp4', r'.')"
del output\*\_tmp_audio16k.wav 2>nul

echo [4/4] Caching MediaPipe model...
python -c "from forge.detect import run_detection; run_detection(r'I:\Nvidia\videos\test.mp4', r'.', labels=['person'], sample_step=15)"

echo Models cached. Ready to zip.
pause