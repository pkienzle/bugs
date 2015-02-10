#!/usr/bin/env python
"""

Model grammar
=============

(MODEL, [statement, ...])
    Form: model { statement ... }

    Always and only occurs as the top level node in the parse tree.  Having
    it simplifies processing when walking the tree. *[statement, ...]* is
    a list of ASSIGN, PRIOR and LOOP nodes.

(LOOP, variable, start, stop, [statement, ...])
    Form: for (variable in start:stop) { statement ... }

    Loop statement.  *variable* must be a simple identifier with no
    subscript indexing. *start* and *stop* are complete expressions.
    *[statement, ...]* is a list of ASSIGN, PRIOR and LOOP nodes.

(ASSIGN, variable, expression)
    Form: variable <- expression

    Usually the *variable* for the ASSIGN node is a variable or an indexed
    variable, but it can also look like a function call.  If it is a variable,
    it will be a DEREF, otherwise it will be an APPLY whose first argument is
    a DEREF.

    For the function-like forms, the APPLY may be shorthand for
    *variable <- inverse fn(expression)*.   OpenBUGS allows *log*, *logit*,
    *cloglog* and *probit* in this context.  For the special case of
    *F(variable)* the APPLY is a functional, where the *expression* right
    hand side represents a function for use in numerical integration and
    root finding.  For the special case of *D(C[k], t)* the APPLY is an ode
    component, where the *expression* right hand side represents the kth
    component of a system of ordinary differential equations.

(PRIOR, variable, distribution, bounds)
    Form: variable ~ distribution bounds

    Identify prior distribution for a variable.  This is represented in
    the language as a distribution function with shape parameters, possibly
    followed by a bounds term with limit expressions.  The bounds
    term node may be (NONE), or it may be an APPLY term with (NONE) parameters.
    The bounds function name seems to be one of *I*, *C* for censored or
    *T* for truncated.

expression
    Form: APPLY, BINOP, UNARY, DEREF, or CONST

    There is no single form for an expression since it may be a bare constant
    or variable, an arithmetic tree or a function call.

(APPLY, name, [argument, ...])
    Form: name(argument, ...)

    If the form appears as part of an expression then the arguments may
    themselves be full expressions, but if the form appears on the left
    hand side of an assignment, then the arguments are restricted to
    simple DEREF nodes.   The interpretation of APPLY depends on
    the name and the context.  As part of an expression it refers to
    a function call.  As the left hand side of an ASSIGN it may
    be an inverse function, a functional or an ode component.  As the right
    hand side of a PRIOR it will be the distribution or the bounds on the
    distribution.  Arguments to bounds functions may be NONE, indicating
    an unspecified bound.

    Reusing APPLY in this way keeps the grammar simpler but doesn't
    significantly complicate the semantics since the special cases all
    occur as direct children of ASSIGN and PRIOR where the content of
    the parent node is required for their interpretation.

(BINOP, op, left, right)
    Form: left op right

    Arithmetic operation (+,-,*,/) combining left and right sub-expressions.
    The grammar enforces the usual precedence rules.

(UNARY, op, term)
    Form: +/- expression

    Negate a term in an expression, or insist that it is positive.

(DEREF, variable, [subscript, ...])
    Form: variable OR variable[subscript, ...]

    Refer to a variable at the particular index.  This may as part of an
    expression, a PRIOR or an ASSIGN. Subscripts may be SLICE, expressions
    or NONE if empty.

(CONST, value)
    Form:  integer OR float OR NaN

    Numeric constant as part of an expression.

(SLICE, start, stop)
    Form: start : stop

    SLICE only occurs as a subscript in a DEREF node, and represent a vector
    slice into the corresponding variable along that index.  *start* and
    *stop* can be expressions. If start or stop is not present, then NONE
    will be used.

(NONE)
    Form:

    Empty expression, which can occur as a subscript, an empty part of a
    slice, as the bound term for unbounded distribution, or as an indefinite
    limit within the bounds definition for a bounded distribution.

"""
from __future__ import print_function

__all__ = ["parse_bugs_model","parse_bugs_rdata","parse_bugs_array",
           "load","loads", "walk","pretty",
           "variables", "observed", "fitted"
           "boundv", "freev", "datav",
           ]

import numpy as np
from pyparsing import (Literal, Optional, Regex, Empty, Keyword,
    ZeroOrMore, OneOrMore, Forward, StringEnd, Word, alphas, alphanums)

###############################################################################
# Parsers
###############################################################################

# === constant ===
def float_or_int(s):
    try: return int(s)
    except: return float(s)
real = Regex("[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][-+]?[0-9]+)?")
real.setParseAction(lambda s,l,t: [float_or_int(t[0])])
NaN = Literal("NA").setParseAction(lambda s,l,t: [np.NaN])
constant = (real|NaN).setName('const')

# === variable ===
#variable =  Regex("[A-Za-z_.][0-9A-Za-z_.]*")
variable = Word(alphas+"_.",alphanums+"_.").setName('var')

CONST = "CONST"
SLICE = "SLICE"
DEREF = "DEREF"
BINOP = "BINOP"
UNARY = "UNARY"
APPLY = "APPLY"
ASSIGN = "ASSIGN"
PRIOR = "PRIOR"
LOOP = "LOOP"
NONE = "NONE"
MODEL = "MODEL"

def _binop(t):
    return t[0] if len(t)==1 else _binop([(BINOP, t[1], t[0], t[2])] + t[3:])
def model_grammar():
    """
    Construct a parser for winBugs/openBugs/JAGS models.

    Returns *parser*, which is a *pyaparsing.ParserElement, with the
    *parser.parserString()* method.

    Be sure to strip comments from the string prior to parsing, so that the
    grammar can be a little simpler.
    """
    ###factor = Forward().setName('factor') # for right-to-left parsing
    expression = Forward().setName('expr')
    group = Forward().setName('group')

    # start:stop used for indexing and for loops
    inner_range = Optional (expression + Optional(Literal(":") + expression))
    paren_range = Literal("(").suppress() + inner_range + Literal(")").suppress()
    slice =  inner_range | paren_range

    paren_range.setName('(slice)')
    inner_range.setName('slice').setParseAction(
        lambda s,l,t: [(SLICE,(NONE,),(NONE,)) if len(t)==0 else (SLICE,t[0],t[2]) if len(t)>1 else t[0]])

    # indexing
    subscripts = slice + ZeroOrMore(Literal(',').suppress() + slice)
    index = Literal('[').suppress() + subscripts + Literal(']').suppress()
    indexed_variable = variable + Optional(index)

    subscripts.setName('subs')
    index.setName('index')
    indexed_variable.setName('deref').setParseAction(
        lambda s,l,t: [(DEREF,t[0],t[1:])])

    # arithmetic
    muldiv = Literal('*') | Literal('/')
    addsub = Literal('+') | Literal('-')
    ###exponent = Literal('^')

    pars = expression + ZeroOrMore(Literal(',').suppress() + expression)
    function = variable + Literal('(') + Optional(pars) + Literal(')')
    paren = Literal('(') + expression + Literal(')')
    value = constant + Empty()
    atom = Optional("-") + (function | indexed_variable | value | paren )
    ###factor << (atom | ZeroOrMore (exponent + factor))
    ###term = factor + ZeroOrMore(muldiv + factor)
    term = atom + ZeroOrMore(muldiv + atom)
    expression << (term + ZeroOrMore(addsub + term))

    paren.setName('paren').setParseAction(
        lambda s,l,t: [t[1]])
    value.setName('value').setParseAction(
        lambda s,l,t: [(CONST, t[0])])
    function.setName('apply').setParseAction(
        lambda s,l,t: [(APPLY, t[0], t[2:-1])])
    atom.setName('atom').setParseAction(
        lambda s,l,t: [(UNARY, t[0], t[1]) if len(t)>1 else t[0]])
    ###factor.setName('factor').setParseAction(lambda s,l,t: [_binop(t)])
    term.setName('term').setParseAction(
        lambda s,l,t: [_binop(t)])
    expression.setName('expr').setParseAction(
        lambda s,l,t: [_binop(t)])

    # priors look like dname(p1,p2,...) with optional qualifier T(left,right)
    # to set the bounds on the prior.   Since left/right are optional, the
    # function parser can't be used, and we need a special bounds term to
    # parse this form.
    bounds_limit = expression | Empty()
    bounds_function = (variable + Literal('(') + bounds_limit
                           + Literal(',') + bounds_limit + Literal(')'))
    bounds = bounds_function | Empty()
    bounds_limit.setName('limit').setParseAction(
        lambda s,l,t: [(t[0] if len(t) else (NONE,))])
    bounds.setName('trunc').setParseAction(
        lambda s,l,t: [(APPLY,t[0],[t[2],t[4]]) if len(t) else (NONE,)])

    # Funky assignment lhs functions, such as:
    #    logit(t) <- alpha; D(C[5],t) <- PER1 * C[7] - R*kT1*C[1]
    lhs_function = variable + Literal('(') + indexed_variable \
                   + ZeroOrMore(Literal(',') + indexed_variable) + Literal(')')
    lhs_function.setName('f(lhs)').setParseAction(
        lambda s,l,t: [(APPLY, t[0], t[2::2])])

    # statements
    assignment = (lhs_function | indexed_variable) + Literal("<-") + expression
    prior = indexed_variable + Literal("~") + function + bounds
    loop = (Keyword("for")  + Literal("(") + variable + Keyword("in")
            + expression + Literal(":") + expression + Literal(")") + group)

    assignment.setName('assign').setParseAction(
        lambda s,l,t: [(ASSIGN, t[0], t[2])])
    prior.setName('prior').setParseAction(
        lambda s,l,t: [(PRIOR, t[0], t[2], t[3])])
    loop.setName('loop').setParseAction(
        lambda s,l,t: [(LOOP, t[2], t[4], t[6], t[8:])])

    # Note: line breaks are ignored.  That means the following are valid:
    #     statement\n
    #     statement;\n
    #     statement; statement\n
    #     statement statement\n
    #     partial statement\n statement completion
    # Indeed, all of these forms exist in the openBugs example models.
    #comment = Literal("#[^\n]*")
    statement = (loop | assignment | prior) + Optional(Literal(';')).suppress()
    body = ZeroOrMore(statement)
    group << (Literal("{").suppress() + body + Literal("}").suppress())
    model = Keyword("model") + group + StringEnd()

    model.setName('model').setParseAction(
        lambda s,l,t: [(MODEL, t[1:])])

    if 0: # Debug
        all_terms = (
            model, group, body, statement, loop, prior, assignment,
            expression, term, atom, muldiv, addsub,
            value, constant, paren, function, pars,
            indexed_variable, variable, index, subscripts,
            slice, inner_range, paren_range,
        )
        for s in all_terms: s.setDebug(True)
    return model

def _build_array(content):
    if len(content.keys()) != 2 or '.Data' not in content or '.Dim' not in content:
        raise ValueError("expected structure to contain .Data and .Dim")
    return np.reshape(content['.Data'],content['.Dim'])

def rdata_grammar():
    pairs = Forward()

    comma_separated_values = constant + ZeroOrMore(Literal(",").suppress() + constant)
    vector = Literal("c(") + comma_separated_values + Literal(")").setName(') to close c')
    vector.setParseAction(lambda s,l,t: [np.array(t[1:-1])])
    array = Literal("structure(") + pairs + Literal(")").setName(') to close structure')
    array.setParseAction(lambda s,l,t: [_build_array(t[1])])
    value = constant | vector | array
    pair = variable + Literal("=").suppress() + value
    pair.setParseAction(lambda s,l,t: [t[:]])
    pairs << (pair + ZeroOrMore(Literal(",").suppress() + pair))
    pairs.setParseAction(lambda s,l,t: [dict(t[:])])
    data = Literal("list(") + pairs  + Literal(")").setName(') to close list')
    data.setParseAction(lambda s,l,t: [t[1]])
    return data


model_parser = model_grammar()
rdata_parser = rdata_grammar()

def _strip_comments(s):
    """
    Strip comments from the data file.

    Leave blank lines so that parser returns the correct line number for errors.
    """
    if 0: # Debug info
        print("   ","1234567890"*7)
        for i,L in enumerate(s.split("\n")): print("% 3d"%(i+1),L)
    return "\n".join(line.split('#',1)[0] for line in s.split('\n'))

def parse_bugs_model(s):
    """
    Parses a bugs model definition file.

    Returns the parse tree.
    """
    s = _strip_comments(s)
    tree = model_parser.parseString(s)
    return tree[0]

def _convert_rdata_value(item):
    if isinstance(item, dict):
        return _convert_rdata_dict(item)
    elif isinstance(item, list):
        return np.array(float(c.replace('NA','NaN')) for c in item)
    else:
        return float(item.replace('NA','NaN'))

def _convert_rdata_dict(pairs):
    pairs = dict((k,_convert_rdata_value(v)) for k,v in pairs.items())
    if ".Data" in pairs:
        return np.reshape(pairs[".Data"], pairs[".Dim"])
    else:
        return pairs

def parse_bugs_rdata(s):
    """
    Parses a bugs data file, with data stored in R format.

    Returns a dictionary of key,value pairs, with *c()* returned as 1-D
    numpy arrays and *structure(.Data=c(),.Dim=c())* returned as n-D array
    of shape *.Dim*.  Strings are not supported.
    """
    s = _strip_comments(s)
    tree = rdata_parser.parseString(s)
    # return _convert_rdata(tree)
    return tree[0]

def parse_bugs_array(s):
    """
    Parses a bugs data array file.

    The file should have column names in the first non-empty line, with each
    name followed by [], a set of rows, with NA for missing data in a column,
    and END on the last non-empty line.

    Comments are introduced by hash and go to the end of the line.  They can
    occur on any line.  Empty lines, and lines including only comments are
    ignored.
    """
    lines = [trimmed
             for line in s.split('\n')
             for trimmed in [line.split('#',1)[0].strip()]
             if trimmed != '']
    header = lines[0]
    data = [L.replace("NA","NaN") for L in lines[1:-1]]
    end = lines[-1]
    column_names = header.split()
    # Check that column names end in []
    assert all(name.endswith('[]') for name in column_names), \
        "invalid column in %r"%header
    # Strip [] from column names
    column_names = [word[:-2] for word in header.split()]
    assert end == "END", "END not at end of data"
    columns = np.loadtxt(data).T
    return dict((k,v) for k,v in zip(column_names, columns))

def loads(s):
    """
    Parse a bugs file.

    Returns *(type, value)* where *type* is "model", "data" or "text" and
    *value* is the corresponding parse tree if it is a model, the data
    dictionary if it is data, or the text of the file if it is neither
    model nor data.

    If you know the file structure, you can call :func:`parse_bugs_model`,
    :func:`parse_bugs_rdata` or :func:`pars_bugs_array` as appropriate.
    """
    maybe_model = False
    for line in s.split('\n'):
        clean =  line.split('#',1)[0].strip()
        if clean == "":
            continue
        if maybe_model:
            if clean.startswith('{'):
                return "model", parse_bugs_model(s)
            else:
                return "text", s
        if clean.startswith('list('):
            return "data", parse_bugs_rdata(s)
        if clean == "model":
            maybe_model = True
            continue
        words = line.split()
        if words[0].startswith("model{"):
            return "model", parse_bugs_model(s)
        if words[0]=="model" and words[1].startswith('{'):
            return "model", parse_bugs_model(s)
        if all(w.endswith('[]') for w in words):
            return "data", parse_bugs_array(s)
        return "text",s

def load(filename):
    with open(filename, 'rt') as fid:
        return loads(fid.read())


##############################################################################
# Parse tree operations
##############################################################################
def _walk_list(slist, parent):
    #print("list",parent[0])
    for i, si in enumerate(slist):
        #print("item",i,statement)
        for d in _walk(si, parent, i): yield d

def _walk(s, parent=None, role=None):
    #print("statement",s[0])
    yield s, parent, role
    if s[0] == SLICE:
        for d in _walk(s[1], s, 'start'): yield d
        for d in _walk(s[2], s, 'stop'): yield d
    elif s[0] == DEREF:
        for d in _walk_list(s[2],s[2]): yield d
    elif s[0] == APPLY:
        for d in _walk_list(s[2],s[2]): yield d
    elif s[0] == BINOP:
        for d in _walk(s[2], s, 'left'): yield d
        for d in _walk(s[3], s, 'right'): yield d
    elif s[0] == UNARY:
        for d in _walk(s[2], s, 'term'): yield d
    elif s[0] == CONST:
        pass
    elif s[0] == ASSIGN:
        for d in _walk(s[1], s, 'variable'): yield d
        for d in _walk(s[2], s, 'expression'): yield d
    elif s[0] == PRIOR:
        for d in _walk(s[1], s, 'variable'): yield d
        for d in _walk(s[2], s, 'distribution'): yield d
        for d in _walk(s[3], s, 'bounds'): yield d
    elif s[0] == LOOP:
        #for d in _walk_statement(s[1], 'variable', s[1]): yield d
        for d in _walk(s[2], s, 'start'): yield d
        for d in _walk(s[3], s, 'stop'): yield d
        for d in _walk_list(s[4], s): yield d
    elif s[0] == NONE:
        pass
    elif s[0] == MODEL:
        for d in _walk_list(s[1], s): yield d
    else:
        raise ValueError("Node %r not recognized in %s:%s"%(s[0],parent[0],note))

def walk(s, context=False):
    """
    Walk the parse tree for the bugs model.

    If *context* is False (the default), then only the node is returned.

    If *context* is True, each element in the sequence yields a
    *(node, parent, role)* tuple. The role names match the names in
    the model grammar, with item number as the role.  For example, the
    third statement in a loop will have a role of 2 (which is the third
    number when counting from 0), where as start and stop will have roles
    of 'start' and 'stop' respectively.  Non-nodes, such as names and
    constants, are not walked.

    The traversal is depth-first.
    """
    if context:
        for context in _walk(s): yield context
    else:
        for node,_,_ in _walk(s): yield node

_PRECEDENCE = { ' ': 0, '+': 1, '-': 1, '*': 2, '/': 2 }
def pretty(s, indent=0, op=' '):
    """
    Pretty prints the bugs model parse tree, returning a formatted string.
    """
    if s[0] == SLICE:
        return pretty(s[1]) + ":" + pretty(s[2])
    elif s[0] == DEREF:
        if len(s[2]) > 0:
            return s[1] + "[" + ", ".join(pretty(si) for si in s[2]) + "]"
        else:
            return s[1]
    elif s[0] == APPLY:
        return s[1] + "(" + ", ".join(pretty(si) for si in s[2]) + ")"
    elif s[0] == BINOP:
        expr = [pretty(s[2],op=s[1]), " ", s[1], " ", pretty(s[3],op=s[1])]
        if _PRECEDENCE[s[1]] < _PRECEDENCE[op]:
            expr = ["("] + expr + [")"]
        return "".join(expr)
    elif s[0] == UNARY:
        return s[1]+"("+pretty(s[2])+")"
    elif s[0] == CONST:
        return str(s[1])
    elif s[0] == ASSIGN:
        return " "*indent + pretty(s[1]) + " <- " + pretty(s[2]) + "\n"
    elif s[0] == PRIOR:
        return " "*indent + pretty(s[1]) + " ~ " + pretty(s[2]) + pretty(s[3]) + "\n"
    elif s[0] == LOOP:
        return " "*indent + "for (" + s[1] + " in " + pretty(s[2]) + " : " + pretty(s[3]) + ") {\n" + "".join(pretty(si, indent+2) for si in s[4]) + " "*indent + "}\n"
    elif s[0] == NONE:
        return ""
    elif s[0] == MODEL:
        return " "*indent + "model {\n"+"".join(pretty(si,indent+2) for si in s[1])+" "*indent+"}\n"
    else:
        return "\n"+" "*indent + "<?%s>\n"%(s[0],)
        #raise ValueError("Node %r not recognized"%(s[0]))

def variables(model):
    """
    Returns the set of variables defined or used in the bugs model.
    """
    return set(node[1] for node in walk(model) if node[0] in (DEREF,LOOP))

def loopv(model):
    """
    Returns the set of looping variables.

    This does not try to exclude temporary variables used within the loop
    but otherwise ignored.
    """
    return set(node[1] for node in walk(model) if node[0] == LOOP)

def boundv(model):
    """
    Return the set of variables defined by the bugs model in assignment
    statements.
    """
    # We are checking if the variable node is an APPLY node, in which
    # case we want the variable name in the first variable of the APPLY
    # node rather than variable name in the ASSIGN node.
    #    (ASSIGN (DEREF identifier [index1, index2, ...]))
    #    (ASSIGN (APPLY fn [(DEREF identifier [index1, index2, ...]), ...]))
    # Don't know what the D(C[4], t) represents in the Sixcomp ode model.
    return set((node[1][1] if node[1][0] == DEREF else node[1][2][0][1])
               for node in walk(model)
               #for _ in [print("boundv",node)]
               if node[0] == ASSIGN) - loopv(model)

def freev(model):
    """
    Returns the set variables that represent fittable parameters.
    """
    return set(node[1][1] for node in walk(model) if node[0] == PRIOR)

def unboundv(model):
    """
    Returns the set of variables that are not bound and not free.

    The undefined variables must be supplied by the data.
    """
    return variables(model) - boundv(model) - freev(model) - loopv(model)


def observed(model, data):
    """
    Set of observed variables, which have an associated distribution and
    a preset value.

    The value presets may either come from the model or the data.
    """
    return freev(model) & (boundv(model) | set(data.keys()))

def fitted(model, data):
    """
    Set of fitted parameters.
    """
    return freev(model) - observed(model, data)

def functions(model):
    """
    Returns all functions and distributions used in the bugs model.

    This is a list not a set so that we can determine usage frequency with
    a simple shell script::

        bugs.py -f *model.txt | grep -v "^===" | sort | uniq -c
    """
    return (node[1] for node in walk(model) if node[0] == APPLY)

def _data_format(v):
    shape = getattr(v, 'shape', [])
    if len(shape) > 1:
        return "\n"+str(v)
    else:
        return str(v)

def pretty_data(data):
    """
    Pretty print a data dictionary.
    """
    return "\n".join("%s = %s"%(k,_data_format(v)) for k,v in sorted(data.items()))

###############################################################################
# Testing
###############################################################################
def _check_match(value, target):
    if isinstance(value, np.ndarray):
        np.testing.assert_equal(value, target)
    elif isinstance(value,dict):
        assert set(value.keys()) == set(target.keys())
        for k,v in value.items():
            _check_match(v,target[k])
    else:
        #print("type(value)",type(value))
        assert value == target, "\nExpected: %s\nObserved: %s"%(target,value)

def test_array():
    target = {"first":np.array((1,3.5e-6)),"second":np.array((2.,np.NaN))}
    source = """\
first[] second[]
1       2
3.5e-6  NA
END
"""
    source_commented = """\
# comment before header
first[] second[] # comment in header
 1\t2 # tabbed line that does not start at first column
# commented out line
3.5e-6  NA
END
# comment after end
"""
    _check_match(parse_bugs_array(source),target)
    _check_match(parse_bugs_array(source_commented),target)

def test_rdata():
    target = {"N":5,"v.b":np.array([1.,2.,np.NaN]),"A":np.array([[1,2],[3,4],[5,6]])}
    source = "list(N=5,v.b=c(1.,2.,NA),A=structure(.Data=c(1,2,3,4,5,6),.Dim=c(3,2)))"
    source_commented = """\
# comment before data
     list( # comment on list line
        N=5, # comment within list
        v.b =\t
c(1.  ,  2., NA ) # comment before comma (weird)
    , A= # comment after equal
    structure(
      .Dim = c(3,2),
      .Data = c(1,2,\t3,4,5,6)
      # comment before close paren
      )

# comment after blank line
\t#comment after mostly blank line
) # comment after data

# a few more comments
# at the end
"""
    _check_match(parse_bugs_rdata(source),target)
    _check_match(parse_bugs_rdata(source_commented),target)

def _check_model(target,source):
    _check_match(parse_bugs_model(source),(MODEL, target))

def test_model():
    _check_model([],"model {}")
    _check_model(
        [(ASSIGN, (DEREF, 'tau.c', []), (CONST, 0.1))],
        "model {tau.c <- 0.1}"
    )
    _check_model(
        [(ASSIGN, (DEREF, 'tau.c', []), (CONST, 0.1))],
        "model {tau.c <- 0.1;}"
    )
    _check_model(
        [(PRIOR,
          (DEREF, 'tau.c', []),
          (APPLY, 'dgamma', [(CONST, 0.1), (CONST, 0.1)]),
          (NONE,))],
        "model {tau.c ~ dgamma(0.1,0.1)}"
    )
    _check_model(
        [(PRIOR,
          (DEREF, 'tau.c', []),
          (APPLY, 'dgamma', [(CONST, 0.1), (CONST, 0.1)]),
          (APPLY, 'T', [(NONE,), (CONST, 0)]))],
        "model {tau.c ~ dgamma(0.1,0.1) T(,0)}"
    )
    _check_model(
        [(PRIOR,
          (DEREF, 'tau.c', []),
          (APPLY, 'dgamma', [(CONST, 0.1), (CONST, 0.1)]),
          (APPLY, 'T', [(NONE,), (CONST, 0)])),
         (ASSIGN,
          (DEREF, 's', []), (BINOP, "/", (CONST, 1), (DEREF, "tau.c", [])))
        ],
        "model {tau.c ~ dgamma(0.1,0.1) T(,0) s <- 1/tau.c}"
    )
    _check_model(
        [(LOOP, 'tau.c', (CONST, 1), (DEREF,'N', []), [])],
        "model {for(tau.c in 1 : N) {}}"
    )
    _check_model(
        [(ASSIGN, (DEREF, 'mu', [(DEREF, 'i', [])]), (CONST, 3))],
        "model {mu[i] <- 3}"
    )
    _check_model(
        [(ASSIGN,
          (DEREF, 'mu', [(DEREF, 'i', []), (DEREF, 'j', [])]),
          (CONST, 3))],
        "model {mu[i, j] <- 3}"
    )
    _check_model(
        [(ASSIGN,
          (DEREF, 'tau.c', []),
          (BINOP, '/',
           (BINOP, '*',
            (BINOP, '/',
             (CONST, 6),
             (CONST, 2)),
            (CONST, 3)),
           (CONST, 9)))],
        "model {tau.c <- 6/2*3/9}"
    )
    _check_model(
        [(ASSIGN,
          (DEREF, 'tau.c', []),
          (BINOP, '+',
           (DEREF, 'alpha', []),
           (BINOP, '*',
            (DEREF, 'beta', []),
            (BINOP, '-',
             (DEREF, 'x', []),
             (DEREF, 'xbar', [])))))],
        "model {tau.c <- alpha + beta * (x - xbar)}"
    )
    _check_model(
        [(ASSIGN,
          (DEREF, 'tau.c', []),
          (BINOP, '+',
           (DEREF, 'alpha', [(DEREF, 'i', [])]),
           (BINOP, '*',
            (DEREF, 'beta', [(DEREF, 'i', [])]),
            (BINOP, '-',
             (DEREF, 'x', [(DEREF, 'j', [])]),
             (DEREF, 'xbar', [])))))],
        "model {tau.c <- alpha[i] + beta[i] * (x[j] - xbar)}"
    )
    _check_model(
        [(ASSIGN,
          (DEREF, 'm', []),
          (DEREF, "t", [(SLICE, (CONST, 1), (DEREF, 'M', []))]))],
        "model {m<-t[1:M]}"
    )
    _check_model(
        [(ASSIGN,
          (APPLY, 'logit', [(DEREF, 'p', [(DEREF, 'j', [])])]),
          (DEREF, 'theta', [(DEREF, 'i', [])]))],
        "model {logit(p[j]) <- theta[i]}"
    )
    _check_model(
        [(ASSIGN,
          (DEREF, 'm', []),
          (DEREF, "t", [(SLICE, (NONE,), (NONE,)), (SLICE, (NONE,), (NONE,))]))],
        "model {m\n <-\nt[\n , ]\n}\n # and a termination comment"
    )
    _check_model(
        [(ASSIGN,
          (DEREF, 'm', []),
          (APPLY, "f",
           [(DEREF, "t",
             [(SLICE, (NONE,), (NONE,)), (SLICE, (NONE,), (NONE,))])]))],
        "model {m <- f(t[ , ])}"
    )
    _check_model(
        [(ASSIGN,
          (DEREF, 'Sigma', [(SLICE, (CONST, 1), (DEREF, "M", [])),
                            (SLICE, (CONST, 1), (DEREF, "M", []))]),
          (APPLY, 'inverse', [(DEREF, 'Omega', [(SLICE, (NONE,), (NONE,)),
                                                (SLICE, (NONE,), (NONE,))])])),
         ],
        "model {Sigma[1 : M , 1 : M] <- inverse(Omega[ , ])}"
    )
    rats = """
model
{
 	for( i in 1 : N ) {
		for( j in 1 : T ) {
			Y[i , j] ~ dnorm(mu[i , j],tau.c)
			mu[i , j] <- alpha[i] + beta[i] * (x[j] - xbar)
		}
		alpha[i] ~ dnorm(alpha.c,alpha.tau)
		beta[i] ~ dnorm(beta.c,beta.tau)
	}
	tau.c ~ dgamma(0.001,0.001)
	sigma <- 1 / sqrt(tau.c)
	alpha.c ~ dnorm(0.0,1.0E-6)
	alpha.tau ~ dgamma(0.001,0.001)
	beta.c ~ dnorm(0.0,1.0E-6)
	beta.tau ~ dgamma(0.001,0.001)
	alpha0 <- alpha.c - xbar * beta.c
}"""
    parse_bugs_model(rats) # just make sure it parses

###############################################################################
# Main program
###############################################################################

def main():
    import sys
    # join all opt strings together
    opts = "".join(arg[1:] for arg in sys.argv[1:] if arg[0] == '-')
    # grab file list from non-opt parameters
    files = [arg for arg in sys.argv[1:] if arg[0] != '-']
    # check that opts and files are valid
    if len(files) == 0 or any((opt not in 'fsdpbv') for opt in opts):
        print("""\
usage: bugs.py [-opts] file file...

Dump a bugs model or data file.  If the file is in odc format, you
will need to run the odcread utility first.  It is available on
github at:

    https://github.com/gertvv/odcread

Options are as follows:

    -s  pretty print the model or data
    -d  show variables required by the model/defined by the data
    -p  show fitting parameters
    -b  show variables defined by the model
    -v  show all variables, including loop variables
""")
        sys.exit(1)

    if not opts: opts = 's'
    view = 's' in opts
    for f in files:
        try: t,v = load(f)
        except:
            import traceback
            print("=== %s : unknown ==="%f)
            print(traceback.format_exc())
        if len(files) > 1:
            print("=== %s : %s ==="%(f,t))
        if t == "model":
            if view: print(pretty(v))
            if 'f' in opts:
                if view:
                    print("functions:", ", ".join(sorted(set(functions(v)))))
                else:
                    print("\n".join(functions(v)))
            if 'v' in opts:
                print("variables:", ", ".join(sorted(variables(v))))
            if 'b' in opts:
                print("bound variables:", ", ".join(sorted(boundv(v))))
            if 'd' in opts:
                print("data variables:", ", ".join(sorted(datav(v))))
            if 'p' in opts:
                print("parameters:", ", ".join(sorted(freev(v))))
        elif t == "data":
            if view: print(pretty_data(v))
            if 'd' in opts:
                print(", ".join(sorted(v.keys())))
        else:
            if view: print(v)
        #for f in walk(v): print(f)

if __name__ == "__main__":
    #test_array()
    #test_rdata()
    #test_model()
    main()

