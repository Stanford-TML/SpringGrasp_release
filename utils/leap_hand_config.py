import numpy as np

ROBOT_CONFIG = {
    'ee_link_idx': [3,7,11,15],
    
    'ee_link_offset': np.array([[0.0, -0.04, 0.015],
                                [0.0, -0.04, 0.015],
                                [0.0, -0.04, 0.015],
                                [0.0, -0.05, -0.015]]),
    
    'ee_link_name': ['fingertip','fingertip_2','fingertip_3','thumb_fingertip'],
    
    'ref_q': np.array([np.pi/15, -np.pi/6, np.pi/15, np.pi/15,
                np.pi/15, 0.0     , np.pi/15, np.pi/15,
                np.pi/15, np.pi/6 , np.pi/15, np.pi/15,
                np.pi/15, np.pi/6 , np.pi/9, np.pi/9]),

    'collision_links':['fingertip','fingertip_2','fingertip_3','thumb_fingertip'],

    'collision_offsets':np.array([[0.0, -0.04, 0.015],
                                  [0.0, -0.04, 0.015],
                                  [0.0, -0.04, 0.015],
                                  [0.0, -0.05, -0.015]]),
    
    'collision_pairs':[[0, 1], [0, 2], [0, 3],
                      [1,2], [1,3],
                      [2,3]]
}