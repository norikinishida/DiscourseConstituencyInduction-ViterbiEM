import copy

import numpy as np

class TreeSampler(object):

    def __init__(self, sampler_names):
        """
        :type sampler_names: list of str
        """
        assert len(sampler_names) == 3
        assert sampler_names[1] != "RB2"
        assert sampler_names[2] != "RB2"

        self.sampler_names = sampler_names

        self.sampler_s = self.get_sampler(sampler_names[0])
        self.sampler_p = self.get_sampler(sampler_names[1])
        self.sampler_d = self.get_sampler(sampler_names[2])

        if self.sampler_s is not None:
            self.use_sbnds = True
        else:
            self.use_sbnds = False

        if self.sampler_p is not None:
            self.use_pbnds = True
        else:
            self.use_pbnds = False

    def get_sampler(self, sampler_name):
        """
        :type sampler_name: str
        :rtype: BaseTreeSampler
        """
        if sampler_name == "X":
            return None
        elif sampler_name == "BU":
            return BU()
        elif sampler_name == "TD":
            return TD()
        elif sampler_name == "RB":
            return RB()
        elif sampler_name == "LB":
            return LB()
        elif sampler_name == "RB2":
            return RB2()
        else:
            raise ValueError("Invalid sampler_name=%s" % sampler_name)

    def sample(self, sexps, edus, edus_head, sbnds, pbnds):
        """
        :type sexps: list of int
        :type edus: list of list of str
        :type edus_head: list of (str, str, str)
        :type sbnds: list of (int, int), or None
        :type pbnds: list of (int, int), or None
        :rtype: list of str
        """
        # Sentence-level sampling
        if self.use_sbnds:
            assert sbnds is not None
            target_bnds = sbnds
            sexps = self.apply_sampler(
                                    sampler=self.sampler_s,
                                    sexps=sexps,
                                    edus=edus,
                                    edus_head=edus_head,
                                    target_bnds=target_bnds)

        # Paragraph-level sampling
        if self.use_pbnds:
            assert pbnds is not None
            if self.use_sbnds:
                target_bnds = pbnds
            else:
                target_bnds = [(sbnds[b][0],sbnds[e][1]) for b,e in pbnds]
            sexps = self.apply_sampler(
                                    sampler=self.sampler_p,
                                    sexps=sexps,
                                    edus=None,
                                    edus_head=None,
                                    target_bnds=target_bnds)

        # Document-level sampling
        sexp = self.sampler_d.sample(sexps=sexps,
                                     edus=None,
                                     edus_head=None) # list of str

        return sexp


    def apply_sampler(self, sampler, sexps, edus, edus_head, target_bnds):
        """
        :type sampler: BaseTreeSampler
        :type sexps: list of int/str
        :type edus: list of list of str
        :type edus_head: list of (str, str, str)
        :type target_bnds: list of (int, int)
        :rtype: list of str
        """
        new_sexps = [] # list of str
        for begin_i, end_i in target_bnds:
            if begin_i == end_i:
                new_sexp = sexps[begin_i] # int/str
                new_sexp = str(new_sexp)
            else:
                new_sexp = sampler.sample(
                                sexps=sexps[begin_i:end_i+1],
                                edus=edus[begin_i:end_i+1] if edus is not None else None,
                                edus_head=edus_head[begin_i:end_i+1] if edus_head is not None else None) # list of str
                new_sexp = " ".join(new_sexp)
            new_sexps.append(new_sexp)
        return new_sexps

class BU(object):
    """
    Random Bottom-Up
    """

    def __init__(self):
        pass

    def sample(self, sexps, edus, edus_head):
        """
        :type sexps: list of int/str
        :type edus: list of list of str
        :type edus_head: list of (str, str, str)
        :rtype: list of str
        """
        x = copy.deepcopy(sexps)
        while len(x) > 1:
            i = np.random.randint(0, len(x)-1)
            lhs = "%s" % x[i]
            rhs = "%s" % x[i+1]
            merged = "( %s %s )" % (lhs, rhs)
            x[i] = merged
            x.pop(i+1)
        sexp = x[0]
        sexp = sexp.split()
        return sexp

class TD(object):
    """
    Random Top-Down
    """

    def __init__(self):
        pass

    def sample(self, sexps, edus, edus_head):
        """
        :type sexps: list of int/str
        :type edus: list of list of str
        :type edus_head: list of (str, str, str)
        :rtype: list of str
        """
        sexp = self.rec_sample(sexps)
        return sexp.split()

    def rec_sample(self, sexps):
        """
        :type sexps: list of int/str
        :rtype: str
        """
        n = len(sexps)
        if n == 1:
            return sexps[0]
        elif n == 2:
            return "( %s %s )"  % (sexps[0], sexps[1])
        k = np.random.randint(1, n)
        lhs = self.rec_sample(sexps=sexps[0:k])
        rhs = self.rec_sample(sexps=sexps[k:])
        sexp = "( %s %s )" % (lhs, rhs)
        return sexp

class RB(object):
    """
    Right Branching
    """

    def __init__(self):
        pass

    def sample(self, sexps, edus, edus_head):
        """
        :type sexps: list of int/str
        :type edus: list of list of str
        :type edus_head: list of (str, str, str)
        :rtype: list of str
        """
        x = copy.deepcopy(sexps)
        while len(x) > 1:
            lhs = x[-2]
            rhs = x[-1]
            merged = "( %s %s )" % (lhs, rhs)
            x[-2] = merged
            x.pop(-1)
        sexp = x[0]
        sexp = sexp.split()
        return sexp

class LB(object):
    """
    Left Branching
    """

    def __init__(self):
        pass


    def sample(self, sexps, edus, edus_head):
        """
        :type sexps: list of int/str
        :type edus: list of list of str
        :type edus_head: list of (str, str, str)
        :rtype: list of str
        """
        x = copy.deepcopy(sexps)
        while len(x) > 1:
            lhs = x[0]
            rhs = x[1]
            merged = "( %s %s )" % (lhs, rhs)
            x[0] = merged
            x.pop(1)
        sexp = x[0]
        sexp = sexp.split()
        return sexp

class RB2(object):
    """
    Syntax-Aware Right Branching
    """

    def __init__(self):
        self.sampler = RB()

    def sample(self, sexps, edus, edus_head):
        """
        :type sexps: list of int
        :type edus: list of list of str
        :type edus_head: list of (str, str, str)
        :rtype: list of str
        """
        # Find the position of the EDU with the ROOT head
        root_position = None
        for edu_i in range(len(sexps)):
            if edus_head[edu_i][2] == "ROOT":
                root_position = edu_i
                break

        if (root_position is not None): #and (len(sexps) == 3 or len(sexps) == 4):
            if root_position == 0:
                sexp_lhs = []
            elif root_position == 1:
                sexp_lhs = [str(sexps[0])]
            else:
                sexp_lhs = self.sampler.sample(sexps[:root_position], edus=None, edus_head=None)

            if root_position == len(sexps)-1:
                sexp_rhs = [str(sexps[-1])]
            else:
                sexp_rhs = self.sampler.sample(sexps[root_position:], edus=None, edus_head=None)

            sexp = sexp_lhs + sexp_rhs
            if len(sexp_lhs) != 0 and len(sexp_rhs) != 0:
                sexp = ["("] + sexp + [")"]
            return sexp
        # if len(sexps) == 3:
        #     # if edus_head[0][2] == "ROOT":
        #     # if (edus[1][0] in ["--", "-lrb-"]) and (edus[1][-1] in ["--", "-rrb-"]):
        #     #     # B が記号で囲まれる
        #     #     sexp = "( ( %d %d ) %d )" % (sexps[0], sexps[1], sexps[2])
        #     #     sexp = sexp.split()
        #     #     return sexp
        #     if edus_head[2][2] == "ROOT":
        #         # C が ROOT
        #         sexp = "( ( %d %d ) %d )" % (sexps[0], sexps[1], sexps[2])
        #         sexp = sexp.split()
        #         return sexp
        #     # elif (edus[0][0] in ["--", "-lrb-"]) and (edus[1][-1] in ["--", "-rrb-"]):
        #     #     # ( A B ) が記号で囲まれる
        #     #     sexp = "( ( %d %d ) %d )" % (sexps[0], sexps[1], sexps[2])
        #     #     sexp = sexp.split()
        #     #     return sexp
        #     # elif edus_head[1][2] == "ROOT":
        #     #     if edus[0][-1] != ",":
        #     #         # print("B が ROOT かつ、A が ',' で終わらない")
        #     #         sexp = "( ( %d %d ) %d )" % (sexps[0], sexps[1], sexps[2])
        #     #         sexp = sexp.split()
        #     #         return sexp
        #     # その他
        #     sexp = "( %d ( %d %d ) )" % (sexps[0], sexps[1], sexps[2])
        #     sexp = sexp.split()
        #     return sexp
        # elif len(sexps) == 4:
        #     # if (edus[1][0] in ["--", "-lrb-"]) and (edus[1][-1] in ["--", "-rrb-"]):
        #     #     sexp = "( ( %d %d ) ( %d %d ) )" % (sexps[0], sexps[1], sexps[2], sexps[3])
        #     #     sexp = sexp.split()
        #     #     return sexp
        #     # elif (edus[2][0] in ["--", "-lrb-"]) and (edus[2][-1] in ["--", "-rrb-"]):
        #     #     sexp = "( %d ( ( %d %d ) %d ) )" % (sexps[0], sexps[1], sexps[2], sexps[3])
        #     #     sexp = sexp.split()
        #     #     return sexp
        #     if edus_head[2][2] == "ROOT":
        #         sexp = "( ( %d %d ) ( %d %d ) )" % (sexps[0], sexps[1], sexps[2], sexps[3])
        #         sexp = sexp.split()
        #         return sexp
        #     elif edus_head[3][2] == "ROOT":
        #         sexp = "( ( %d ( %d %d ) ) %d )" % (sexps[0], sexps[1], sexps[2], sexps[3])
        #         sexp = sexp.split()
        #         return sexp
        #     # その他
        #     sexp = "( %d ( %d ( %d %d ) ) )" % (sexps[0], sexps[1], sexps[2], sexps[3])
        #     sexp = sexp.split()
        #     return sexp
        else:
            sexp = self.sampler.sample(sexps, edus, edus_head)
            # x = copy.deepcopy(sexps)
            # while len(x) > 1:
            #     lhs = x[-2]
            #     rhs = x[-1]
            #     merged = "( %s %s )" % (lhs, rhs)
            #     x[-2] = merged
            #     x.pop(-1)
            # sexp = x[0]
            # sexp = sexp.split()
            return sexp


