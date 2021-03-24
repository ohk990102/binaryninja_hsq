import abc
import struct
from collections import OrderedDict
from typing import List, Optional


class Instr(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, addr: int, data: List[int]):
        self.addr = addr
        self.data = data
        self.width = len(data)
        self.operands = []

    def get_next_addr(self) -> List[int]:
        if self.c not in [-1, 0, self.addr + self.width]:
            return [self.addr + self.width, self.c]
        return [self.addr + self.width]

    def __str__(self):
        ret = f"{type(self).__name__.lower()}"
        for operand in self.operands:
            ret += f" {operand}"
        return ret

    def get_human_friendly_str(self, registers):
        ret = f"{type(self).__name__.lower()}"
        for operand in self.operands:
            if operand in registers.keys():
                ret += f" {registers[operand]}"
            else:
                ret += f" {operand}"
        return ret


class Subleq(Instr):
    def __init__(self, addr: int, data: List[int]):
        super().__init__(addr, data)
        self.a, self.b, self.c = self.data
        self.operands = [self.a, self.b, self.c]


class Clear(Instr):
    def __init__(self, addr: int, data: List[int]):
        super().__init__(addr, data)
        self.a, self.b, self.c = self.data
        self.operands = [self.a, self.c]

    def get_next_addr(self) -> List[int]:
        if self.c not in [-1, 0]:
            return [self.c]
        return []


class Exit(Instr):
    def get_next_addr(self) -> List[int]:
        return []


class Jmp(Instr):
    def __init__(self, addr: int, data: List[int]):
        super().__init__(addr, data)
        self.c = self.data[2]
        self.operands = [self.c]

    def get_next_addr(self) -> List[int]:
        return [self.c]


class Mov(Instr):
    def __init__(self, addr: int, data: List[int]):
        super().__init__(addr, data)
        self.a = self.data[3]
        self.b = self.data[4]
        self.c = self.data[5]
        self.operands = [self.a, self.b, self.c]


class Dec(Instr):
    def __init__(self, addr: int, data: List[int]):
        super().__init__(addr, data)
        self.a, self.b, self.c = self.data
        self.operands = [self.b, self.c]

    def get_next_addr(self) -> List[int]:
        if self.c not in [-1, 0, self.addr + self.width]:
            return [self.addr + self.width, self.c]
        return [self.addr + self.width]


class Inc(Instr):
    def __init__(self, addr: int, data: List[int]):
        super().__init__(addr, data)
        self.b = self.data[1]
        self.operands = [self.b]

    def get_next_addr(self) -> List[int]:
        return [self.addr + self.width]


class Call(Instr):
    def __init__(self, addr: int, data: List[int]):
        super().__init__(addr, data)
        self.c = data[26]
        self.operands = [self.c]

    def get_next_addr(self) -> List[int]:
        if self.c not in [-1, 0]:
            return [self.addr + self.width, self.c]
        return [self.addr + self.width]


class Ret(Instr):
    def get_next_addr(self) -> List[int]:
        return []


class Push(Instr):
    def __init__(self, addr: int, data: List[int]):
        super().__init__(addr, data)
        self.v = data[24]
        self.c = data[26]
        self.operands = [self.v, self.c]

    def get_next_addr(self) -> List[int]:
        if self.c not in [-1, 0, self.addr + self.width]:
            return [self.addr + self.width, self.c]
        return [self.addr + self.width]


class Pop(Instr):
    def __init__(self, addr: int, data: List[int]):
        super().__init__(addr, data)
        self.v = data[10]
        self.c = data[14]
        self.operands = [self.v, self.c]

    def get_next_addr(self) -> List[int]:
        if self.c not in [-1, 0, self.addr + self.width]:
            return [self.addr + self.width, self.c]
        return [self.addr + self.width]


rules = OrderedDict()
rules[Clear] = {"pattern": [(Subleq, "A", "A", None)], "width": 3}
rules[Jmp] = {"pattern": [(Clear, "Z", None)], "width": 3}
rules[Exit] = {"pattern": [(Jmp, -1)], "width": 3}
rules[Dec] = {"pattern": [(Subleq, "dec", None, None)], "width": 3}
rules[Inc] = {"pattern": [(Subleq, "inc", None, None)], "width": 3}
rules[Mov] = {
    "pattern": [(Clear, "A", "$"), (Subleq, None, "A", None)],
    "width": 6,
}
rules[Push] = {
    "pattern": [
        (Dec, "sp", "$"),
        (Mov, "sp", "$+6", "$"),
        (Mov, "sp", "$+1", "$"),
        (Clear, 0, "$"),
        (Mov, "sp", "$+1", "$"),
        (Subleq, None, 0, None),
    ],
    "width": 27,
}
rules[Pop] = {
    "pattern": [(Mov, "sp", "$+3", "$"), (Mov, 0, None, "$"), (Inc, "sp", "$")],
    "width": 15,
}
rules[Call] = {
    "pattern": [
        (Push, "$", None),
    ],
    "width": 28,
}
rules[Ret] = {
    "pattern": [(Mov, "sp", "$+3", "$"), (Mov, 0, "$+2", "$"), (Jmp, 0)],
    "width": 15,
}

keywords = ["inc", "Z", "dec", "ax", "bp", "sp"]


class HsqDisassembler(object):
    def __init__(self, data: bytes, int_size: int):
        self.int_size = int_size
        if self.int_size not in (4, 8):
            raise Exception("Invalid int size. ")
        self.instr_align = 3
        self.data = list(
            map(
                lambda v: v[0],
                struct.iter_unpack("<i" if self.int_size == 4 else "<q", data),
            )
        )
        self.instrs: list[Optional[Instr]] = [None for _ in range(len(self.data))]
        self.tabs = -1
        self._phase1()
        self._phase2()

    def get_instr(self, addr: int):
        if self.is_valid_addr(addr):
            return self.instrs[addr]
        return None

    def put_instr(self, addr: int, instr: Instr):
        self.instrs[addr] = instr
        for i in range(addr + 1, addr + instr.width):
            self.instrs[i] = None

    def gen_simple_instr(self, addr: int):
        if not self.is_valid_addr(addr, 3):
            return None

        data = self.data[addr : addr + 3]
        instr = Subleq(addr, data)
        return instr

    def _phase1(self):
        # Heuristics on getting register info
        top = self.gen_simple_instr(0)
        if not (top.a == 0 and top.b == 0):
            raise Exception("Does not match higher subleq arch")
        self.sqmain = top.get_next_addr()[1]
        instr: Subleq = self.gen_simple_instr(self.sqmain)
        self.inc = instr.a - 2
        self.Z = instr.a - 1
        self.dec = instr.a
        self.ax = instr.a + 1
        self.bp = instr.a + 2
        self.sp = instr.a + 3
        self.symbol = {
            self.inc: "inc",
            self.Z: "Z",
            self.dec: "dec",
            self.ax: "ax",
            self.bp: "bp",
            self.sp: "sp",
        }

    def _phase2(self):
        visited = set()
        queue = set([0])

        while len(queue) != 0:
            addr = queue.pop()
            if addr in visited:
                continue
            visited.add(addr)
            instr = self.put_best_instr(addr)
            if instr is not None:
                for addr in instr.get_next_addr():
                    queue.add(addr)

    def put_best_instr(self, addr: int) -> Optional[Instr]:
        instr = self.get_instr(addr)
        if instr is not None:
            return instr
        instr = self.gen_simple_instr(addr)
        if instr is None:
            return None
        self.tabs += 1
        print("\t" * self.tabs + f"{addr}: [None] -> {str(instr)}")

        self.put_instr(addr, instr)
        for key, rule in rules.items():
            if self.peek_and_test_pattern(addr, rule["pattern"]):
                print("\t" * self.tabs + f"{addr}: {str(instr)} -> ", end="")
                instr = key(addr, self.data[addr : addr + rule["width"]])
                print(f"{str(instr)}")
                self.put_instr(addr, instr)
        self.tabs -= 1
        return instr

    def peek_and_test_pattern(self, addr: int, pattern: List[tuple]) -> bool:
        cu = addr
        cache = {}
        for i in range(len(pattern)):
            part = pattern[i]
            instr = self.get_instr(cu)
            if instr is None:
                instr = self.put_best_instr(cu)
                if instr is None:
                    return False

            if not isinstance(instr, part[0]):
                return False
            for j in range(len(instr.operands)):
                if part[j + 1] is None:
                    continue
                elif isinstance(part[j + 1], int):
                    if part[j + 1] != instr.operands[j]:
                        return False
                elif part[j + 1].startswith("$"):
                    offset = 0 if part[j + 1][1:] == "" else int(part[j + 1][1:])

                    if cu + instr.width + offset != instr.operands[j]:
                        return False
                elif part[j + 1] in keywords:
                    if getattr(self, part[j + 1]) != instr.operands[j]:
                        return False
                else:
                    if part[j + 1] in cache.keys():
                        if cache[part[j + 1]] != instr.operands[j]:
                            return False
                    else:
                        cache[part[j + 1]] = instr.operands[j]
            next = instr.get_next_addr()
            if i == len(pattern) - 1:
                break
            if len(next) != 1:
                return False
            cu = next[0]
        return True

    def is_valid_addr(self, addr: int, width: int = 1) -> bool:
        return 0 <= addr < len(self.data) - (width - 1)

    def is_register(self, addr: int) -> bool:
        return addr in self.symbol.keys() and self.symbol[addr] in ["sp", "bp", "ax"]

    def is_fixed_symbol(self, addr: int) -> bool:
        return addr in self.symbol.keys() and self.symbol[addr] in ["inc", "Z", "dec"]

    def __str__(self):
        ret = ""
        for i in range(len(self.instrs)):
            if self.instrs[i] is not None:
                ret += f"{i}: {self.instrs[i].get_human_friendly_str(self.symbol)}\n"
        return ret


if __name__ == "__main__":
    with open("./test2.bin", "rb") as f:
        data = f.read()
    disassembler = HsqDisassembler(data, 8)
    # print(disassembler.data)
    print(disassembler)
    print(disassembler.data[disassembler.inc])
    print(disassembler.data[disassembler.Z])
    print(disassembler.data[disassembler.dec])
    print(disassembler.data[disassembler.ax])
    print(disassembler.data[disassembler.bp])
    print(disassembler.data[disassembler.sp])
