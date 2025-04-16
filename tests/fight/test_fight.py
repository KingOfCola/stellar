# -*-coding:utf-8 -*-
"""
@File      :   test_fight.py
@Time      :   2025/04/16 17:56:40
@Author    :   Urvan Christen
@Version   :   1.0
@Contact   :   urvan.christen@gmail.com
@Desc      :   Tests of the fight module.
"""

from fight import fight


def test_fib():
    assert fight.fib(0) == 0
    assert fight.fib(1) == 1
    assert fight.fib(2) == 1
    assert fight.fib(3) == 2
    assert fight.fib(4) == 3
    assert fight.fib(5) == 5
    assert fight.fib(6) == 8
    assert fight.fib(7) == 13
    assert fight.fib(8) == 21
    assert fight.fib(9) == 34
    assert fight.fib(10) == 55
