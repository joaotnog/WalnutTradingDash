import unittest
from Functions import *
import streamlit as st
from BacktestZone import *

 
class TestRun(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        a = 1
        self.data = data
        # from BacktestZone import data, st
        # self.st, self.data, self.start_date, self.end_date = st, data, start_date, end_date       
        
    def data_loaded(self):
        """ensures that the expected columns are all present"""
        self.assertTrue(self.data.shape[0]>10) 
        self.assertTrue(self.data.shape[1]==5)
    
    def run_indicator(self):
        indicator = 'Simple Moving Average (SMA)'
        function_indicator = implement_simple_moving_average
        # entry_comparator, exit_comparator, entry_data1, entry_data2, exit_data1, exit_data2 = implement_simple_moving_average(self.st, self.data, self.start_date, self.end_date)
        
# self = TestRun()        
 
TestRun.main()