NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# 1. Pre Looper (Video / Pixel Pipeline)
try:
    from .nodes_pre_looper import PreLooperMinimaxNode
    NODE_CLASS_MAPPINGS["PreLooperMinimax"] = PreLooperMinimaxNode
    NODE_DISPLAY_NAME_MAPPINGS["PreLooperMinimax"] = "Pre Looper (MiniMax H3)"
except Exception as e:
    print(f"[LooperMinimaxH3] Error loading PreLooperMinimax: {e}")
    import traceback
    traceback.print_exc()

# 2. Post Looper Video (Video / Pixel Pipeline)
try:
    from .nodes_post_looper import PostLooperVideoMinimaxNode
    NODE_CLASS_MAPPINGS["PostLooperVideoMinimax"] = PostLooperVideoMinimaxNode
    NODE_DISPLAY_NAME_MAPPINGS["PostLooperVideoMinimax"] = "Post Looper Video (MiniMax H3)"
except Exception as e:
    print(f"[LooperMinimaxH3] Error loading PostLooperVideoMinimax: {e}")
    import traceback
    traceback.print_exc()

# 3. Pre Looper Latent & Preview (Native Latent Pipeline)
try:
    from .nodes_pre_looper_latent import PreLooperLatentMinimaxNode, PreLooperLatentPreviewMinimaxNode
    NODE_CLASS_MAPPINGS["PreLooperLatentMinimax"] = PreLooperLatentMinimaxNode
    NODE_DISPLAY_NAME_MAPPINGS["PreLooperLatentMinimax"] = "Pre Looper Latent (MiniMax H3)"

    NODE_CLASS_MAPPINGS["PreLooperLatentPreviewMinimax"] = PreLooperLatentPreviewMinimaxNode
    NODE_DISPLAY_NAME_MAPPINGS["PreLooperLatentPreviewMinimax"] = "Pre Looper Latent Preview (MiniMax H3)"
except Exception as e:
    print(f"[LooperMinimaxH3] Error loading PreLooperLatentMinimax: {e}")
    import traceback
    traceback.print_exc()

# 4. Post Looper Latent (Native Latent Pipeline)
try:
    from .nodes_post_looper_latent import PostLooperLatentMinimaxNode
    NODE_CLASS_MAPPINGS["PostLooperLatentMinimax"] = PostLooperLatentMinimaxNode
    NODE_DISPLAY_NAME_MAPPINGS["PostLooperLatentMinimax"] = "Post Looper Latent (MiniMax H3)"
except Exception as e:
    print(f"[LooperMinimaxH3] Error loading PostLooperLatentMinimax: {e}")
    import traceback
    traceback.print_exc()

# 5. Server routes for interactive timeline player
try:
    from .server_routes import setup_server_routes
    setup_server_routes()
except Exception:
    pass

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
