import os
import json
import subprocess
import shutil
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS

app = Flask(__name__, static_folder=None)
CORS(app)

# Configuration
BASE_DIR = Path('/app/data/ai-stitching-model')
UPLOAD_DIR = BASE_DIR / 'dataset/sems/manual'
PROGRESS_FILE = BASE_DIR / 'progress_pipeline.json'

# Ensure upload directory exists
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

@app.route('/')
def index():
    print(f"Index requested. Serving from {BASE_DIR}")
    return send_from_directory(BASE_DIR, 'progress.html')

@app.route('/<path:path>')
def serve_static(path):
    full_path = BASE_DIR / path
    print(f"Request: {path}, Full path: {full_path}, Exists: {full_path.exists()}")
    return send_from_directory(BASE_DIR, path)

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'files[]' not in request.files:
        return jsonify({'error': 'No files provided'}), 400
    
    files = request.files.getlist('files[]')
    saved_files = []
    
    # Clear existing manual uploads if new batch comes in? 
    # For now, let's keep adding, or maybe user wants to clear.
    # Let's assume user wants to process what they upload.
    # To be safe for a "batch" run, maybe we should clear 'manual' first?
    # Let's just append for now, user can manage via file names or we can add a clear trigger.
    
    for file in files:
        if file.filename == '':
            continue
        
        if file and (file.filename.endswith('.png') or file.filename.endswith('.jpg')):
            filepath = UPLOAD_DIR / file.filename
            file.save(filepath)
            saved_files.append(file.filename)
    
    return jsonify({
        'message': f'Successfully uploaded {len(saved_files)} files',
        'files': saved_files
    })

@app.route('/cleanup', methods=['POST'])
def cleanup_files():
    try:
        # Clear upload directory
        if UPLOAD_DIR.exists():
            for f in UPLOAD_DIR.glob('*'):
                if f.is_file():
                    f.unlink()
        
        # Also clear output directories to be safe (already done in run, but good here too)
        mask_dir = BASE_DIR / 'dataset/masks/manual'
        vector_dir = BASE_DIR / 'dataset/vectorized_manual'
        for d in [mask_dir, vector_dir]:
            if d.exists():
                for f in d.glob('*'):
                    try: 
                        if f.is_file(): f.unlink()
                    except: pass
        
        # Clear status files
        for f in BASE_DIR.glob('progress_*.json'):
            try: f.unlink()
            except: pass
                    
        return jsonify({'message': 'Cleanup successful'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/run', methods=['POST'])
def run_pipeline():
    try:
        # Check if we have manual files
        manual_files = list(UPLOAD_DIR.glob('*.png')) + list(UPLOAD_DIR.glob('*.jpg'))
        if not manual_files:
            return jsonify({'error': 'No files in manual upload directory to process'}), 400

        # CLEAR STALE DATA (and status)
        mask_dir = BASE_DIR / 'dataset/masks/manual'
        vector_dir = BASE_DIR / 'dataset/vectorized_manual'
        
        for d in [mask_dir, vector_dir]:
            if d.exists():
                for f in d.glob('*'):
                    try: 
                        if f.is_file(): f.unlink()
                    except: pass

        for f in BASE_DIR.glob('progress_*.json'):
            try: f.unlink()
            except: pass
            
        # Initialize status to 0%
        import json
        with open(BASE_DIR / 'progress_pipeline.json', 'w') as f:
            json.dump({
                "stage": "Ready", 
                "status": "Initializing pipeline...", 
                "percent": 0
            }, f)

        # Construct command
        cmd = [
            'python3', 'run_stitching.py',
            '--vectorize',
            '--show-labels',
            '--show-borders',
            '--use-manual-dir'
        ]
        
        # Run in background
        subprocess.Popen(cmd)
        
        return jsonify({'message': 'Pipeline started successfully'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/manual_panorama.gds')
def download_gds_file():
    try:
        file_path = BASE_DIR / 'manual_panorama.gds'
        if not file_path.exists():
            return jsonify({'error': 'File not found'}), 404
            
        return send_file(file_path, as_attachment=True, download_name='manual_panorama.gds', mimetype='application/octet-stream', max_age=0)
    except Exception as e:
        return jsonify({'error': f'Error sending file: {e}'}), 500

@app.route('/download/svg')
def download_svg():
    try:
        return send_from_directory(BASE_DIR, 'manual_panorama.svg', as_attachment=True)
    except Exception as e:
        return jsonify({'error': f'File not found: {e}'}), 404

@app.route('/status')
def get_status():
    if PROGRESS_FILE.exists():
        try:
            with open(PROGRESS_FILE, 'r') as f:
                data = json.load(f)
            return jsonify(data)
        except:
            return jsonify({'status': 'Error reading status'})
    else:
        return jsonify({'status': 'Idle', 'stage': 'Ready'})

if __name__ == '__main__':
    print("Starting server on port 8000...")
    app.run(host='0.0.0.0', port=8000)
