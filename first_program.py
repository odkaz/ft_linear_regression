import ft_linear_regression
import json

def read_json(url):
    with open(url, 'r') as f:
        data = json.load(f)
    return data

def main():
    milage = 0
    t0, t1 = 0, 0

    while True:
        try:
            milage = float(input("Please enter the milage: "))
            break
        except ValueError:
            print("Oops!  That was no valid number.  Try again...")

    try:
        data = read_json('train.json')
        t0, t1 = data['t0'], data['t1']
    except FileNotFoundError:
        pass

    price = ft_linear_regression.estimate_price(t0, t1, milage)
    print('estimated price is:', price)

if __name__ == '__main__':
    main()
