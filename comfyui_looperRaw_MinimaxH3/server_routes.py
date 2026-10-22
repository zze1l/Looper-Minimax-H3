import os

ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".webm", ".mkv", ".gif", ".flv", ".wmv"}

def setup_server_routes():
    """Registers API routes on ComfyUI server for timeline metadata and video upload."""
    try:
        import server
        prompt_server = getattr(server, "PromptServer", None)
        if prompt_server is None:
            return
        instance = getattr(prompt_server, "instance", None)
        if instance is None or not hasattr(instance, "routes"):
            return
        
        from aiohttp import web
        import folder_paths
        from .looper_utils import get_video_metadata, get_video_path
        routes = instance.routes

        @routes.get("/looper_h3/video_info")
        async def get_info_handler(request):
            raw_filename = request.query.get("filename", "")
            if not raw_filename:
                return web.json_response({"error": "Parameter 'filename' required"}, status=400)
            
            filename = os.path.basename(raw_filename.replace("\\", "/"))
            video_path = get_video_path(filename)
            meta = get_video_metadata(video_path)
            meta["filename"] = filename
            meta["found"] = os.path.isfile(video_path)
            return web.json_response(meta)

        @routes.post("/looper_h3/upload_video")
        async def upload_video_handler(request):
            try:
                reader = await request.post()
                video_data = reader.get("video") or reader.get("file")
                if not video_data:
                    return web.json_response({"error": "No video file provided"}, status=400)
                
                raw_filename = getattr(video_data, "filename", "") or "uploaded_video.mp4"
                filename = os.path.basename(raw_filename.replace("\\", "/"))
                if not filename:
                    filename = "uploaded_video.mp4"

                ext = os.path.splitext(filename)[1].lower()
                if ext not in ALLOWED_VIDEO_EXTENSIONS:
                    return web.json_response(
                        {"error": f"Invalid video extension '{ext}'. Allowed: {sorted(list(ALLOWED_VIDEO_EXTENSIONS))}"},
                        status=400
                    )

                input_dir = folder_paths.get_input_directory()
                os.makedirs(input_dir, exist_ok=True)

                target_path = os.path.join(input_dir, filename)
                
                # Write uploaded bytes to input directory in 64KB chunks to prevent RAM spikes
                total_bytes = 0
                with open(target_path, "wb") as f:
                    while True:
                        chunk = video_data.file.read(65536)
                        if not chunk:
                            break
                        f.write(chunk)
                        total_bytes += len(chunk)
                
                meta = get_video_metadata(target_path)
                print(f"[LooperMinimaxH3] Video uploaded successfully: {filename} ({total_bytes} bytes)")
                
                return web.json_response({
                    "success": True,
                    "name": filename,
                    "filename": filename,
                    "metadata": meta
                })
            except Exception as e:
                print(f"[LooperMinimaxH3] Error uploading video: {e}")
                return web.json_response({"error": str(e)}, status=500)

        print("[LooperMinimaxH3] API routes registered successfully (/looper_h3/video_info & /looper_h3/upload_video)")
    except Exception as e:
        print(f"[LooperMinimaxH3] Info: Server route registration skipped ({e}). Native HTML5 player will manage metadata.")
