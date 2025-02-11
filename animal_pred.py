import tkinter as tk
from tkinter import messagebox, filedialog
from lib import detUtils
import os, cv2, datetime, sys,pygame,threading
from PIL import Image, ImageTk, ImageDraw, ImageFont
import numpy as np
from datetime import datetime
from bytetrack.mc_bytetrack import MultiClassByteTrack
import tkinter.font as tkFont
try:
    import winsound
except ImportError:
    # If not on Windows, handle sound using system beep
    def play_system_beep():
        print("\a")

class Colors:
    def __init__(self):
        hexs = (
            "FF3838","FF9D97","FF701F","FFB21D","CFD231","48F90A","92CC17","3DDB86","1A9334","00D4BB",
            "2C99A8","00C2FF","344593","6473FF","0018EC","8438FF","520085","CB38FF","FF95C8","FF37C7",)
        self.palette = [self.hex2rgb(f"#{c}") for c in hexs]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):
        return tuple(int(h[1 + i : 1 + i + 2], 16) for i in (0, 2, 4))

class WebcamPredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Webcam Predictor")
        self.root.geometry("720x960")
        
        # Enable the window to resize
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        #------------------------------------------------------#
        # (A) Create a font and a label to display log entries
        #------------------------------------------------------#
        self.bold_font = tkFont.Font(size=14, weight="bold")  # Customize size, weight, etc.

        self.create_log_window()
        # This will store recent log messages. You can limit how many lines you display.
        self.log_messages = []

        # Create a StringVar to manage the state of radio buttons
        self.log_save_option = tk.StringVar(value="save")  # Default: "save"

        # Radio button: Save photos with log
        self.save_radio = tk.Radiobutton(
            root,
            text="SAVE FRAME(LOG)",
            variable=self.log_save_option,
            value="save"
        )
        self.save_radio.grid(row=0, column=0, padx=1, pady=1, sticky="w")

        # Radio button: Exclude photos from log
        self.exclude_radio = tk.Radiobutton(
            root,
            text="WITHOUT SAVING FRAME(LOG)",
            variable=self.log_save_option,
            value="exclude"
        )
        self.exclude_radio.grid(row=0, column=1, padx=1, pady=1, sticky="w")

        self.sound_played = False
        pygame.mixer.init() 
        self.warning_sound_path = 'lib/warningsound.wav'
        self.sound_enabled = False
        self.animals_visualized = False
        # Initialize video capture as None (will be set later)
        self.cap = None
        self.colors = Colors()
        self.model = detUtils.DetUtils('lib/best640_640.onnx', conf_thres=0.4, iou_thres=0.3)
        
        self.load_video_button = tk.Button(root, text="Load MP4 File", command=self.load_mp4_file)
        self.load_video_button.grid(row=7, column=0, columnspan=2, padx=10, pady=10, sticky="ew")

        slider_frame = tk.Frame(root, bg="yellow")
        slider_frame.grid(row=8, column=0, columnspan=2, pady=10, sticky="ew")

        self.current_time_label = tk.Label(slider_frame, text="00:00", width=8, anchor="e", bg="yellow")
        self.current_time_label.pack(side="left", padx=5)

        self.slider = tk.Scale(slider_frame, from_=0, to=100, orient="horizontal", length=400, command=self.on_slider_move)
        self.slider.pack(side="left", fill="x", expand=True)

        self.total_time_label = tk.Label(slider_frame, text="00:00", width=8, anchor="w", bg="yellow")
        self.total_time_label.pack(side="right", padx=5)

        self.slider_active = False
        self.slider.config(state='disabled')

        self.video_mode = None  # Possible values: 'webcam', 'rtsp', 'mp4'
        self.mp4_file_path = None

        self.pause_button = tk.Button(root, text="pause", command=self.toggle_pause)
        self.pause_button.grid(row=9, column=0, columnspan=2, padx=10, pady=10, sticky="ew")
        self.is_paused = False  # State to track pause status
        self.pause_button.config(state='disabled')

        # UI components
        self.label = tk.Label(root, text="Please enter the RTSP URL. If the URL is blank, the WEBCAM image is automatically played.")
        self.label.grid(row=1, column=0, columnspan=2, pady=10, sticky="ew")

        # Text entry for RTSP URL
        self.rtsp_entry = tk.Entry(root)
        self.rtsp_entry.grid(row=2, column=0, columnspan=2, pady=10, sticky="ew")

        # Button to start the video stream
        self.start_button = tk.Button(root, text="Start Video", command=self.start_video)
        self.start_button.grid(row=3, column=0, padx=5, pady=10, sticky="ew")

        # Button to stop the video stream (next to the start button)
        self.stop_button = tk.Button(root, text="Stop Video", command=self.stop_video)
        self.stop_button.grid(row=3, column=1, padx=5, pady=10, sticky="ew")

        # Canvas for displaying video
        self.canvas = tk.Canvas(root)
        self.canvas.grid(row=4, column=0, columnspan=2, padx=10, pady=(0, 5), sticky="nsew")

        self.image_on_canvas = None  # Placeholder for image on the canvas

        # Button for toggling sound
        self.sound_button = tk.Button(root, text="Enable Sound", command=self.toggle_sound)
        self.sound_button.grid(row=6, column=0, columnspan=2, padx=10, pady=10, sticky="ew")
        self.class_name = ["roe deer", "wild boar", "chipmunk", "squirrel", "water deer"]
        # Configure row and column stretching behavior
        self.root.grid_rowconfigure(4, weight=8)
        self.root.grid_columnconfigure(0, weight=1)  
        self.root.grid_columnconfigure(1, weight=1)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.tracker = MultiClassByteTrack(
            fps=30,  # set a default frame rate
            track_thresh=0.5,
            track_buffer=30,
            match_thresh=0.3,
            min_box_area=10,
            mot20=True,
        )

        if not os.path.exists("ToolLogs"):
            os.makedirs("ToolLogs")
        

        self.track_id_status = {} 
        self.track_id_dict = {}

    def create_log_window(self):

        self.log_window = tk.Toplevel(self.root)
        self.log_window.title("Command Log Window")
        self.log_window.configure(bg="black")
        self.log_window.geometry("500x300")

        self.log_text = tk.Text(self.log_window, bg="black", fg="#00FF00", font=("Courier", 15))
        self.log_text.pack(expand=True, fill="both")

    def update_logs(self, message):

        self.log_text.delete("1.0", tk.END) 
        self.log_text.insert(tk.END, message)  
        self.log_text.see(tk.END)          

    def on_slider_move(self, value):
        """ Handle user interaction with the slider to seek video position. """
        if self.cap is not None and self.video_mode == 'mp4':
            self.slider_active = True
            

            frame_position = int(value)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_position)

            fps = self.cap.get(cv2.CAP_PROP_FPS)

            current_time = frame_position / fps

            current_minutes = int(current_time // 60)
            current_seconds = int(current_time % 60)

            self.current_time_label.config(text=f"{current_minutes:02d}:{current_seconds:02d}")

            self.slider_active = False

    def toggle_pause(self):
        """ Toggle pause/resume state for MP4 playback. """
        if self.video_mode == 'mp4':
            self.is_paused = not self.is_paused
            if self.is_paused:
                self.pause_button.config(text="play")
            else:
                self.pause_button.config(text="pause")
                self.update_frame()

    def load_mp4_file(self):
        """ Open a file dialog to select an MP4 file and start video processing. """
        self.stop_video()  # Ensure any ongoing processing is stopped before starting a new mode

        self.mp4_file_path = filedialog.askopenfilename(
            filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")]
        )
        if not self.mp4_file_path:
            return  # User cancelled file selection

        # Set up video capture for the MP4 file
        self.cap = cv2.VideoCapture(self.mp4_file_path)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Failed to open MP4 file.")
            return

        # Set the slider's maximum value based on video duration (in frames)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_time = self.total_frames / fps

        total_minutes = int(self.total_time // 60)
        total_seconds = int(self.total_time % 60)
        self.total_time_label.config(text=f"{total_minutes:02d}:{total_seconds:02d}")

        self.slider.config(to=self.total_frames - 1, state='normal')  # Enable the slider for MP4 files

        # Indicate that the app is in MP4 video mode
        self.video_mode = 'mp4'
        self.is_paused = False  # Ensure processing starts when the video is loaded
        self.pause_button.config(text="pause")
        self.pause_button.config(state='normal')  # Enable the Pause button for MP4 mode

        # Reset the video position to the beginning
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.slider.set(0)  # Reset the slider to the start position
        self.current_time_label.config(text="00:00")  # Reset current time label
        self.root.update()
        # Start the video processing loop
        self.update_frame()
        
    def reset_app_state(self):
        """ Reset the application state for a fresh start in a new mode. """
        # Stop any ongoing frame updates
        self.track_id_dict = {}
        self.is_paused = True  # Pause any ongoing video playback
        self.video_mode = None  # Clear the current mode

        # Release any existing video capture resources
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        # Reset UI elements
        self.pause_button.config(state='disabled', text="Pause")
        self.slider.config(state='disabled')
        self.slider.set(0)

        # Clear the canvas
        if self.image_on_canvas is not None:
            self.canvas.delete(self.image_on_canvas)
            self.image_on_canvas = None

        # Stop any playing sounds (if applicable)
        self.stop_warning_sound()

    def start_video(self):
        """ Start the video stream based on RTSP URL or webcam if empty. """
        self.stop_video()  # Ensure any ongoing processing is stopped before starting a new mode
        self.reset_app_state()  # Reset state before starting a new mode

        rtsp_url = self.rtsp_entry.get().strip()
        self.cap = cv2.VideoCapture(0 if rtsp_url == "" else rtsp_url)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Failed to open video stream.")
            return

        # Indicate the mode ('webcam' or 'rtsp')
        self.video_mode = 'webcam' if rtsp_url == "" else 'rtsp'
        self.slider.config(state='disabled')  # Disable the slider for webcam and RTSP modes
        self.pause_button.config(state='disabled')  # Disable the Pause button for webcam and RTSP modes

        # Start the video processing loop
        self.is_paused = False  # Ensure processing starts when the video stream begins
        self.update_frame()

    def stop_video(self):
        self.reset_app_state()
        """ Stop the video stream and all processing. """
        if self.cap is not None:
            self.cap.release()  # Release the video capture
            self.cap = None  # Set cap to None to stop the update loop

        # Clear the canvas by removing the last image
        if self.image_on_canvas is not None:
            self.canvas.delete(self.image_on_canvas)  # Delete the image from the canvas
            self.image_on_canvas = None  # Reset the image reference

        # Stop any ongoing AI processing
        self.is_paused = True
        self.video_mode = None  # Clear video mode to prevent further processing
        self.stop_warning_sound() 

    def update_frame(self):
        """ Process and display video frames based on the current mode. """
        if self.cap is None or self.is_paused:
            # If cap is None or the video is paused/stopped, stop updating frames
            return

        ret, frame = self.cap.read()
        if not ret:
            if self.video_mode == 'mp4':
                # For MP4 files, stop playback at the end of the file
                self.stop_video()
            return

        if self.video_mode == 'mp4' and not self.slider_active:
            # Get the current position of the video (in frames) and update the slider
            current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            self.slider.set(current_frame)

            current_time = current_frame / self.cap.get(cv2.CAP_PROP_FPS)
            minutes = int(current_time // 60)
            seconds = int(current_time % 60)
            self.current_time_label.config(text=f"{minutes:02d}:{seconds:02d}")

        # Run detection on the resized frame (only if not paused)
        if not self.is_paused:
            boxes, scores, classes = self.model.detect_objects(frame)
            t_ids, t_bboxes, t_scores, t_class_ids = self.tracker(frame, boxes, scores, classes)
            
            logs_for_frame = []
            
            if len(classes) == 0:
                logs_for_frame.append("")
            else :
                for lbox, lscore, lcls in zip(boxes,scores,classes):

                    ui_text = f"Animal_Log > Detected - {self.class_name[lcls]} {lscore*100}"
                    
                    logs_for_frame.append(ui_text)

            combined_logs = "\n".join(logs_for_frame)
            self.update_logs(combined_logs)
            
            for tracker_id, bbox, cls in zip(t_ids, t_bboxes, t_class_ids):

                if tracker_id not in self.track_id_dict:
                    new_id = len(self.track_id_dict)
                    self.track_id_dict[tracker_id] = new_id

                if tracker_id not in self.track_id_status:
                    self.track_id_status[tracker_id] = {"status": "active", "class_name": self.class_name[cls]}
                    self.log_event(self.track_id_dict[tracker_id], "IN", self.class_name[cls],frame)
                elif self.track_id_status[tracker_id]["status"] == 'inactive':
                    self.track_id_status[tracker_id]["status"] = 'active'
                    self.track_id_status[tracker_id]["class_name"] = self.class_name[cls]

            active_ids = set(t_ids)

            for tracked_id in list(self.track_id_status.keys()):
                if tracked_id not in active_ids and self.track_id_status[tracked_id]["status"] == 'active':
                    self.track_id_status[tracked_id]["status"] = 'inactive'
                    obj_class = self.track_id_status[tracked_id]["class_name"]
                    self.log_event(self.track_id_dict.get(tracked_id, tracked_id), "OUT", obj_class,frame)

            if len(classes) > 0:

                self.animals_visualized = True
                frame_with_boxes = self.detect_draw_box(frame, t_ids, t_bboxes, t_class_ids)
            else:
                frame_with_boxes = frame
                self.animals_visualized = False

            if self.sound_enabled and self.animals_visualized:
                self.play_warning_sound()
            else:
                self.stop_warning_sound()

        else:
            # If paused, just display the frame without running detection
            frame_with_boxes = frame
        # Get the current canvas size
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        # Resize the frame to fit the canvas
        frame_resized = cv2.resize(frame_with_boxes, (canvas_width, canvas_height))
        # Convert the frame to a format suitable for Tkinter
        img_pil = Image.fromarray(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))
        img_tk = ImageTk.PhotoImage(img_pil)

        # Update the canvas with the resized frame
        if self.image_on_canvas is None:
            self.image_on_canvas = self.canvas.create_image(0, 0, anchor="nw", image=img_tk)
        else:
            self.canvas.itemconfig(self.image_on_canvas, image=img_tk)
        self.canvas.image = img_tk  # Keep a reference to avoid garbage collection

        # Call update_frame again after a delay (33ms for ~30fps)
        if self.video_mode in ['webcam', 'rtsp', 'mp4']:
            self.root.after(33, self.update_frame)

    def toggle_sound(self):
        self.sound_enabled = not self.sound_enabled
        if self.sound_enabled:
            self.sound_button.config(text="Disable Sound")
        else:
            self.sound_button.config(text="Enable Sound")

    def play_warning_sound(self):
        
        threading.Thread(target=self._play_sound, daemon=True).start()

    def _play_sound(self):
        """Actual sound playing function."""
        if not self.sound_played:
            try:
                pygame.mixer.music.load(self.warning_sound_path)
                pygame.mixer.music.play(-1)  # Play on loop until stopped
                self.sound_played = True
            except NameError:
                play_system_beep()

    def stop_warning_sound(self):
        """Stop the warning sound if it is playing."""
        if self.sound_played:
            pygame.mixer.music.stop()
            self.sound_played = False

    def detect_draw_box(self, image, ids, boxes ,classes):
        img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)

        if hasattr(sys, '_MEIPASS'):
            font_path = os.path.join(sys._MEIPASS, 'lib/NanumGothic-Bold.ttf')
        else:
            font_path = "lib/NanumGothic-Bold.ttf" 
        font = ImageFont.truetype(font_path, 30)

        for box, cls, t_id in zip(boxes, classes, ids):
            x1, y1, x2, y2 = box.astype(int)
            label = f" ID: {self.track_id_dict[t_id]} {self.class_name[cls]}"

            draw.rectangle([x1, y1, x2, y2], outline=self.colors(int(cls), True), width=3)
            draw.text((x1, y1 - 20), label, font=font, fill=self.colors(int(cls), True))

        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    def on_closing(self):

        self.stop_video()

        pygame.mixer.quit()
        if hasattr(self, "log_window"):
            self.log_window.destroy()
        self.root.quit()

    def log_event(self, track_id, event, obj_class, frame=None):

        current_time = datetime.now().strftime('%H:%M:%S')
        log_entry = f"[{current_time}] ID {track_id} {obj_class} {event}\n"
        log_file_path = f"./ToolLogs/{datetime.now().strftime('%Y-%m-%d')}_log.txt"
        with open(log_file_path, "+a", encoding="utf-8") as log_file:
            log_file.write(log_entry)
        #self.update_logs(log_entry)

        if self.log_save_option.get() == "save" and frame is not None:
            os.makedirs("./ToolLogs/SavedFrames", exist_ok=True)

            # Convert the frame to RGB if it is in RGBA mode
            pil_image = Image.fromarray(frame)
            if pil_image.mode == "RGBA":
                pil_image = pil_image.convert("RGB")

            # Save the image as a JPEG file
            frame_output_path = f"./ToolLogs/SavedFrames/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_ID_{track_id}_{event}.jpg"
            pil_image.save(frame_output_path, "JPEG")

# Start the Tkinter application
if __name__ == "__main__":
    root = tk.Tk()
    app = WebcamPredictorApp(root)
    root.mainloop()
