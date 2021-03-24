from binaryninja.architecture import Architecture
from binaryninja.binaryview import BinaryView
from binaryninja.callingconvention import CallingConvention
from binaryninja.enums import BranchType, InstructionTextTokenType, SegmentFlag
from binaryninja.function import InstructionInfo, InstructionTextToken, RegisterInfo
from binaryninja.lowlevelil import LowLevelILFunction, LowLevelILLabel

from subleq64.instructions import *


class HigherSubleq64(Architecture):
    name = "hsq64"
    address_size = 8
    default_int_size = 8
    instr_alignment = 1
    max_instr_length = address_size * 32
    disassembler: HsqDisassembler = None

    regs = {
        "sp": RegisterInfo("sp", 8),
        "bp": RegisterInfo("bp", 8),
        "ax": RegisterInfo("ax", 8),
    }
    stack_pointer = "sp"

    def get_instruction_info(self, data, addr):
        instr = self.disassembler.instrs[addr // self.address_size]
        if instr is None:
            return None

        result = InstructionInfo()
        result.length = instr.width * self.address_size
        next_addr = instr.get_next_addr()
        if isinstance(instr, Call):
            result.add_branch(BranchType.CallDestination, instr.c * self.address_size)
        elif isinstance(instr, Ret) or isinstance(instr, Exit):
            result.add_branch(BranchType.FunctionReturn)
        else:
            if len(next_addr) == 2:
                result.add_branch(BranchType.TrueBranch, next_addr[1] * self.address_size)
                result.add_branch(BranchType.FalseBranch, next_addr[0] * self.address_size)
            elif len(next_addr) == 1:
                result.add_branch(BranchType.UnconditionalBranch, next_addr[0] * self.address_size)
        return result

    def get_instruction_text(self, data, addr):
        instr = self.disassembler.instrs[addr // self.address_size]
        if instr is None:
            return None

        tokens = []
        tokens.append(
            InstructionTextToken(InstructionTextTokenType.TextToken, type(instr).__name__.lower())
        )
        tokens.append(InstructionTextToken(InstructionTextTokenType.TextToken, " "))
        for i in range(len(instr.operands)):
            operand = instr.operands[i]
            if self.disassembler.is_register(operand):
                tokens.append(
                    InstructionTextToken(
                        InstructionTextTokenType.RegisterToken,
                        self.disassembler.symbol[operand],
                    )
                )
            else:
                tokens.append(
                    InstructionTextToken(
                        InstructionTextTokenType.PossibleAddressToken,
                        hex(operand * self.address_size),
                    )
                )

            if i != len(instr.operands) - 1:
                tokens.append(
                    InstructionTextToken(InstructionTextTokenType.OperandSeparatorToken, ", ")
                )

        return tokens, instr.width * self.address_size

    def get_instruction_low_level_il(self, data, addr, il: LowLevelILFunction):
        instr = self.disassembler.instrs[addr // self.address_size]
        if instr is None:
            return None

        if isinstance(instr, Subleq):
            a, b, c = instr.a, instr.b, instr.c

            _, mem_a = self.get_addr_mem_il(a, il)
            addr_b, mem_b = self.get_addr_mem_il(b, il)
            sub_op = il.sub(self.address_size, mem_b, mem_a)
            if self.disassembler.is_register(b):
                store_b = il.set_reg(self.address_size, self.disassembler.symbol[b], sub_op)
            else:
                store_b = il.store(self.address_size, addr_b, sub_op)
            il.append(store_b)
            less_op = il.compare_signed_less_equal(
                self.address_size, mem_b, il.const(self.address_size, 0)
            )

            t_target = il.get_label_for_address(il.arch, c * self.address_size)
            t_label_found = True
            if t_target is None:
                t_label_found = False
                t_target = LowLevelILLabel()

            f_label_found = True
            f_target = il.get_label_for_address(il.arch, addr + instr.width + self.address_size)
            if f_target is None:
                f_target = LowLevelILLabel()
                f_label_found = False

            il.append(il.if_expr(less_op, t_target, f_target))

            if not t_label_found:
                il.mark_label(t_target)
                il.append(il.jump(il.const(self.address_size, c * self.address_size)))
            if not f_label_found:
                il.mark_label(f_target)
        elif isinstance(instr, Clear):
            b = instr.b
            c = instr.c

            addr_b, _ = self.get_addr_mem_il(b, il)
            store_b = il.store(self.address_size, addr_b, il.const(self.address_size, 0))
            il.append(store_b)
            jump_c = il.jump(il.const(self.address_size, instr.c * self.address_size))
            il.append(jump_c)

        elif isinstance(instr, Push):
            v = instr.v

            addr_v, mem_v = self.get_addr_mem_il(v, il)
            push_v = il.push(self.address_size, mem_v)

            il.append(push_v)
        elif isinstance(instr, Mov):
            a, b = instr.a, instr.b

            addr_a, mem_a = self.get_addr_mem_il(a, il)
            addr_b, mem_b = self.get_addr_mem_il(b, il)
            if self.disassembler.is_register(b):
                mov_op = il.set_reg(self.address_size, self.disassembler.symbol[b], mem_a)
            else:
                mov_op = il.store(self.address_size, addr_b, mem_a)
            il.append(mov_op)
        elif isinstance(instr, Ret):
            il.append(il.ret(il.load(self.address_size, il.reg(self.address_size, "sp"))))
        elif isinstance(instr, Pop):
            v = instr.v
            addr_v, _ = self.get_addr_mem_il(v, il)
            pop_op = il.pop(self.address_size)
            if self.disassembler.is_register(v):
                store_op = il.set_reg(self.address_size, self.disassembler.symbol[v], pop_op)
            else:
                store_op = il.store(self.address_size, addr_v, pop_op)
            il.append(store_op)
        elif isinstance(instr, Call):
            il.append(il.call(il.const(self.address_size, instr.c * self.address_size)))
        elif isinstance(instr, Inc):
            b = instr.b
            addr_b, mem_b = self.get_addr_mem_il(b, il)
            if self.disassembler.is_register(b):
                store_op = il.set_reg(
                    self.address_size,
                    self.disassembler.symbol[b],
                    il.add(self.address_size, mem_b, il.const(self.address_size, 1)),
                )
            else:
                store_op = il.store(
                    self.address_size,
                    addr_b,
                    il.add(self.address_size, mem_b, il.const(self.address_size, 1)),
                )
            il.append(store_op)
        elif isinstance(instr, Dec):
            b = instr.b
            addr_b, mem_b = self.get_addr_mem_il(b, il)
            if self.disassembler.is_register(b):
                store_op = il.set_reg(
                    self.address_size,
                    self.disassembler.symbol[b],
                    il.add(self.address_size, mem_b, il.const(self.address_size, -1)),
                )
            else:
                store_op = il.store(
                    self.address_size,
                    addr_b,
                    il.add(self.address_size, mem_b, il.const(self.address_size, -1)),
                )
            il.append(store_op)
        elif isinstance(instr, Exit):
            il.append(il.no_ret())
        elif isinstance(instr, Jmp):
            il.append(il.jump(il.const(self.address_size, instr.c * self.address_size)))

        return instr.width * self.address_size

    def get_addr_mem_il(self, addr, il):
        if self.disassembler.is_register(addr):
            addr_il = il.reg(self.address_size, self.disassembler.symbol[addr])
            mem_il = addr_il
        else:
            addr_il = il.const_pointer(self.address_size, addr * self.address_size)
            mem_il = il.load(self.address_size, addr_il)
        return addr_il, mem_il


HigherSubleq64.register()


class DefaultCallingConvention(CallingConvention):
    name = "default"
    int_arg_regs = []
    int_return_reg = "ax"


arch = Architecture["hsq64"]
arch.register_calling_convention(DefaultCallingConvention(arch, "default"))
arch.standalone_platform.default_calling_convention = arch.calling_conventions["default"]


class HigherSubleqView(BinaryView):
    name = "hsq"
    long_name = "Higher Subleq Binary"

    @classmethod
    def is_valid_for_data(self, data):
        header = data.read(0, 8)
        return header == b"\x00" * 8

    def __init__(self, data):
        BinaryView.__init__(self, parent_view=data, file_metadata=data.file)

        # TODO: allow different bits
        self.data = data
        self.disassembler = HsqDisassembler(self.data[:], 8)

        HigherSubleq64.disassembler = self.disassembler
        self.arch = Architecture["hsq64"]
        self.platform = Architecture["hsq64"].standalone_platform

    def init(self):
        self.add_auto_segment(
            0,
            len(self.data),
            0,
            len(self.data),
            SegmentFlag.SegmentReadable
            | SegmentFlag.SegmentWritable
            | SegmentFlag.SegmentExecutable,
        )
        return True

    def perform_is_executable(self):
        return True

    def perform_get_entry_point(self):
        return 0


HigherSubleqView.register()
