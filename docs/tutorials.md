This part of the project documentation focuses on a **learning-oriented** approach.
You'll learn how to get started with the code in this project.

Expand this section by considering the following points:

- Help newcomers with getting started
- Teach readers about your library by making them write code
- Inspire confidence through examples that work for everyone, repeatably
- Give readers an immediate sense of achievement
- Show concrete examples, no abstractions
- Provide the minimum necessary explanation
- Avoid any distractions

!!! warning
    Faça ao menos 1 tutorial com jupyter notebook.

``` python
>>> from fico.chronologicalsampling import *
>>> data = pd.read_csv(f'data/data.csv')
>>> data
            date    open     low    high   close  adj_close    volume  shares_outstanding
0     2002-12-31   73.71   73.71   75.01   74.09      42.94   8233484        1.804664e+09
1     2003-01-02   75.33   74.75   77.03   77.03      44.64   8226267        1.804664e+09
2     2003-01-03   77.15   76.68   78.06   78.06      45.24   6236566        1.804664e+09
3     2003-01-06   78.30   78.21   81.07   79.91      46.32   8285680        1.804664e+09
4     2003-01-07   80.26   80.07   82.39   82.22      47.65  12454617        1.804664e+09
...          ...     ...     ...     ...     ...        ...       ...                 ...
5089  2023-03-21  126.90  125.66  127.15  126.57     126.57   3839064        9.058000e+08
5090  2023-03-22  127.00  124.01  127.22  124.05     124.05   3514905        9.058000e+08
5091  2023-03-23  123.81  122.60  124.93  123.37     123.37   4643905        9.058000e+08
5092  2023-03-24  123.36  122.88  125.40  125.29     125.29   3809255        9.058000e+08
5093  2023-03-27  126.47  126.47  130.25  129.31     129.31   6498029        9.058000e+08

[5094 rows x 8 columns]
```
