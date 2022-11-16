import numpy as np
from pytracking.evaluation.data import Sequence, BaseDataset, SequenceList


class FishDataset(BaseDataset):
    """
    fishies
    """
    def __init__(self, split=None):
        super().__init__()
        self.base_path = self.env_settings.fish_path
        self.split = split
        self.sequence_list = self._get_sequence_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        sequence_path = sequence_name
        nz = 5
        ext = 'jpg'
        start_frame = 1

        anno_path = '{}/annotations/{}_obj0.txt'.format(self.base_path, sequence_name)
        try:
            ground_truth_rect = np.loadtxt(str(anno_path), dtype=np.float64)
        except:
            ground_truth_rect = np.loadtxt(str(anno_path), delimiter=',', dtype=np.float64)

        end_frame = ground_truth_rect.shape[0]

        frames = ['{base_path}/jpgs/{sequence_path}/{frame:0{nz}}.{ext}'.format(base_path=self.base_path,
                  sequence_path=sequence_path, frame=frame_num, nz=nz, ext=ext)
                  for frame_num in range(start_frame, end_frame+1)]

        # Convert gt
        if ground_truth_rect.shape[1] > 4:
            gt_x_all = ground_truth_rect[:, [0, 2, 4, 6]]
            gt_y_all = ground_truth_rect[:, [1, 3, 5, 7]]

            x1 = np.amin(gt_x_all, 1).reshape(-1,1)
            y1 = np.amin(gt_y_all, 1).reshape(-1,1)
            x2 = np.amax(gt_x_all, 1).reshape(-1,1)
            y2 = np.amax(gt_y_all, 1).reshape(-1,1)

            ground_truth_rect = np.concatenate((x1, y1, x2-x1, y2-y1), 1)
        return Sequence(sequence_name, frames, 'fish', ground_truth_rect)

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        if self.split is not None:
            with open(f'{self.base_path}/attributes/{self.split}_videos.txt') as f:
                sequence_list = f.read().splitlines()
        else:
            with open(f'{self.base_path}/attributes/all_videos.txt') as f:
                sequence_list = f.read().splitlines()

        # sequence_list= ['girdhar_parrotfish_30fps_480p_usvi_GOPR1481',
        #                 'hanlon_octopus_30fps_480p1_short',
        #                 'kukulya_shark_30fps_480p',
        #                 'hanlon_octopus_30fps_480p5_st',
        #                 'hanlon_octopus_30fps_480p6',
        #                 'hanlon_octopus_30fps_480p9',
        #                 'kukulya_shark_bottom_30fps_480p',
        #                 'kukulya_shark_snow_30fps_480p',
        #                 'kukulya_taylor_dolphins_30fps_480p_st',
        #                 'mesobot_larvacean_30fps_480p2',
        #                 'mesobot_larvacean_30fps_480p3',
        #                 'mesobot_solmissus_30fps_480p0',
        #                 'mesobot_solmissus_30fps_480p1',
        #                 'girdhar_blue_tang_30fps_480p_usvi_GOPR1463',
        #                 'girdhar_blue_tang_30fps_480p_usvi_GOPR1473',
        #                 'girdhar_boxfish_30fps_480p_usvi_GOPR1466',
        #                 'girdhar_gray_angelfish_30fps_480p_usvi_GOPR1482',
        #                 'girdhar_lionfish_30fps_480p_usvi_GOPR1455',
        #                 'girdhar_parrotfish_30fps_480p_usvi_GOPR1464',
        #                 'girdhar_parrotfish_30fps_480p_usvi_GOPR1480',
        #                 'girdhar_striped_fish_30fps_480p_usvi_GOPR1472',
        #                 'mooney_reef_squid_solo_30fps_480p',
        #                 'mooney_reef_squid_duo_30fps_480p']

        return sequence_list
