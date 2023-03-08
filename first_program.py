import ft_linear_regression

def main():
    milage = 0
    t0, t1 = 0, 0
    price = ft_linear_regression.estimate_price(t0, t1, milage)
    print('estimated price is:', price)

if __name__ == '__main__':
    main()
