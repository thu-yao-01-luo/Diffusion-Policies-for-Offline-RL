from dataclasses import dataclass 
from utils.utils_zhiao import load_config

@dataclass
class Config:
    name: str = 'zhiao'
    age: int = 18
    id: str = 'student'

def main():
    args = load_config(Config)
    # print(args)
    print(args.name)
    print(args.age)
    print(args.id)

if __name__ == '__main__':
    main()