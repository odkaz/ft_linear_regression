import ft_linear_regression
import json

def output_json(data, url):
    with open(url, 'w') as f:
        json.dump(data, f, indent=2)

def main():
    t0, t1 = ft_linear_regression.linear_regression()
    res = {'t0': t0, 't1': t1}
    output_json(res, 'train.json')

if __name__ == '__main__':
    main()
