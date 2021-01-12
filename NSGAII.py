#!usr/bin/env python
'''
Created on 202101011
@author: think
'''

'''
Created on 20210111
@author: kakamesyli
'''
import random
import operator
class Pop(object):
    def __init__(self,pop_index,x_min,x_max,x_num,f_num,fun_name):
        self.index = pop_index; #个体编号
        self.var = []; #个体变量值
        for i in range(x_num):
            self.cal_var(x_min,x_max);
        self.output = []; #个体输出值
        for j in range(f_num):
            self.cal_output(self.var,x_num,fun_name[j]);
        
    def cal_var(self,x_min,x_max):
        self.var.append(x_min + (x_max-x_min) * random.random());
    
    def cal_output(self,x,x_num,fun_name):
        self.output.append(obj_fun(x,x_num,fun_name));
    
    def create_rank_asm(self,Pop):
        self.rank_asm.append(Pop);
        
class Chromo(object):
    def __init__(self,pop_num,x_min,x_max,x_num,f_num,fun_name):
        self.pop_asm = [];
        self.pareto_rank = 0;
        self.crowding = 0;
        self.dom_num = 0;
        self.dom_asm = [];
        self.rank_asm = []; 
        for i in range(pop_num):
            self.create_pop_asm(self,pop_num,x_min,x_max,x_num,f_num,fun_name);
    
    def create_pop_asm(self,pop_index,x_min,x_max,x_num,f_num,fun_name):
            self.pop.append(Pop(pop_index,x_min,x_max,x_num,f_num,fun_name));
    
        
	def non_dominate_sort(self,pop_num,f_num):
        for i in range(pop_num):
            Chromo.dom_num = 0;
            Chromo.dom_asm = [];
            for j in pop_num:
                less = 0;
                equal = 0;
                greater = 0;
                if (i.value[0] < j.value[0] and i.value[1] < j.value[1]):
                    less = less + 1;
                elif (i.value[0] == j.value[0] and i.value[1] == j.value[1]):
                    equal = equal + 1;
                else:
                    greater = greater + 1;
                if (less == 0 and equal != f_num):
                    Chromo.dom_num = Chromo.dom_num +1;
                elif (greater == 0 and equal != f_num):
                    Chromo.dom_asm.append(j);
                if Chromo.dom_num == 0:
                    Chromo.pareto_rank = 1;
                    Chromo.rank_asm.append(i);
        while (len(Chromo.rank_asm) != 0):
            for pop in Chromo.rank_asm:
                if (len(Chromo.dom_asm) != 0):

def obj_fun(x,x_num,fun_name):
        if operator.eq(fun_name, 'ZDT1'):
            f = [];
            f[0] = x[1];
            s = 0;
            for i in range(x):
                s = s + x[i];
            g = 1 + 9 * (s/(x_num-1));
            f[1] = g * (1-(f[1]/g)^0.5);
        return f;