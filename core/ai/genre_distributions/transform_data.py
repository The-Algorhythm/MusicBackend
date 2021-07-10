import numpy as np
import json
import time


def main():
    start = time.time()
    with open('checkpoint.json', 'r') as f:
        user_distributions_map = json.loads(f.read())
    arr = np.array([x for x in user_distributions_map.values()])
    print(f"Loaded data in {time.time() - start}s")
    np.savetxt('user_distributions_100k.gz', arr)


if __name__ == '__main__':
    main()
