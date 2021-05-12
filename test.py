import unittest
import os
import src.yolohelper
from src.yolohelper import *

class TestYoloHelper(unittest.TestCase):
    
    def setUp(self) -> None:
        self.datapath = os.path.join(os.getcwd(), "data")
        self.results = os.path.join(os.getcwd(),"src", "results")
        self.mockimg = [os.path.join(self.datapath, "im1.jpg")]
        self.mocktext= [os.path.join(self.datapath, "im1.txt")]

    def test_valid_yolo_data_path(self):
        path = is_valid_yolo_data_directory(self.datapath, "Train")
        self.assertEqual(path, self.datapath)
    
    
    def test_prepare_yolo_files_list(self):
        print(ImageFileFormatEnum.jpg)
        datafiles = prepare_yolo_files_list(ImageFileFormatEnum.jpg, self.datapath, "")
        self.assertGreater(len(datafiles), 0)
        self.assertEqual(datafiles[0], os.path.join(self.datapath, "im1.jpg"))

    def test_copy_yolo_files(self):
        flag = copy_yolo_files([os.path.join(self.datapath, "im1.jpg")], self.results)
        self.assertTrue(flag)

    def test_write_yolo_file(self):
        write_yolo_file(self.results, "test.txt", [os.path.join(self.datapath, "im1.jpg")])
        self.assertTrue(os.path.exists(os.path.join(self.results,"test.txt")))

    def test_validate_yolo_input_data(self):
        flag = validate_yolo_input_data(self.datapath,self.results)
        self.assertTrue(flag)

unittest.main(argv=[''],verbosity=0, exit=False)