import time
import winsound


class AlertEngine:

    def __init__(self):

        # Current alert states
        self.phone_alert = False
        self.drowsiness_alert = False

        # Used to prevent the alarm from
        # playing continuously every frame.
        self.last_phone_alert = 0
        self.last_drowsiness_alert = 0

        # Minimum time between repeated sounds
        self.alert_cooldown = 3.0


    # ==========================================
    # PLAY ALERT SOUND
    # ==========================================

    def play_alert(self):

        try:

            # Windows built-in beep
            winsound.Beep(
                1000,
                500
            )

        except Exception:

            # Fallback
            winsound.MessageBeep()


    # ==========================================
    # PHONE ALERT
    # ==========================================

    def trigger_phone_alert(self):

        current_time = time.monotonic()

        self.phone_alert = True

        if (
            current_time - self.last_phone_alert
            >= self.alert_cooldown
        ):

            self.play_alert()

            self.last_phone_alert = current_time


    # ==========================================
    # DROWSINESS ALERT
    # ==========================================

    def trigger_drowsiness_alert(self):

        current_time = time.monotonic()

        self.drowsiness_alert = True

        if (
            current_time - self.last_drowsiness_alert
            >= self.alert_cooldown
        ):

            self.play_alert()

            self.last_drowsiness_alert = current_time


    # ==========================================
    # RESET PHONE ALERT
    # ==========================================

    def reset_phone_alert(self):

        self.phone_alert = False


    # ==========================================
    # RESET DROWSINESS ALERT
    # ==========================================

    def reset_drowsiness_alert(self):

        self.drowsiness_alert = False


    # ==========================================
    # RESET EVERYTHING
    # ==========================================

    def reset(self):

        self.phone_alert = False
        self.drowsiness_alert = False