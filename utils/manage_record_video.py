import cv2
import time


class manage_record_video:

    def __init__(self):
        self.initialed = False
        self.estimate_fps = 30
        self.last_recorded = time.time()

        self.initialed_2 = False
        self.estimate_fps_2 = 30
        self.last_recorded_2 = time.time()
        self.last_delay_frame = 0
        self.th_delay_frame = 150


    def init_video(self, frame):
        if self.initialed == False:
            (frame_height, frame_width) = frame.shape[:2]
            self.idx_saved = self.read_last_saved()
            self.out = cv2.VideoWriter('saved_record/hasil_' + str(self.idx_saved) + '.avi',
                                  cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), self.estimate_fps, (frame_width, frame_height))
            with open('saved_record/last_idx.txt', 'w') as f:  # write increment idx last send
                f.write("%s\n" % (self.idx_saved + 1))

            self.initialed = True

    def init_video_2(self, frame):
        if self.initialed_2 == False:
            (frame_height, frame_width) = frame.shape[:2]
            self.out_2 = cv2.VideoWriter('saved_record/hasil_' + str(self.idx_saved) + '_2.avi',
                                  cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), self.estimate_fps, (frame_width, frame_height))

            self.initialed_2 = True


    def read_last_saved(self):  # Membaca dataset versi ke berapa yang digunakan
        file = open("saved_record/last_idx.txt")
        for line in file:
            fields = line.strip().split()
            return int(fields[0])


    def do_record(self, frame):
        if time.time() - self.last_recorded > (1 / self.estimate_fps):
            self.out.write(frame)
            self.last_recorded = time.time()


    def do_record_2(self, frame, boxes):
        if len(boxes) > 0:
            if time.time() - self.last_recorded_2 > (1 / self.estimate_fps_2):
                self.out_2.write(frame)
                self.last_recorded_2 = time.time()
                self.last_delay_frame = 0

        elif self.last_delay_frame < self.th_delay_frame:
            self.out_2.write(frame)
            self.last_delay_frame += 1


    def do_record_3(self, frame, boxes):
        self.out_2.write(frame)


    def release(self):
        self.out.release()
        self.out_2.release()
