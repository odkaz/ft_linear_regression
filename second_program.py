import ft_linear_regression


def main():
    t0, t1 = ft_linear_regression.linear_regression()
    f = open('res.txt', 'w')
    res = str(t0) + ',' + str(t1)
    f.write(res)
    f.close()

if __name__ == '__main__':
    main()
