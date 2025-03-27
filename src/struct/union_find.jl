mutable struct UnionFind
    parentId::Vector{Int}
    rank::Vector{Int}
end

function makeUnionFind(n::Int)
    return UnionFind(1:n, zeros(n))
end

function find(uf::UnionFind, i::Int)
    if uf.parentId[i] != i
        uf.parentId[i] = find(uf, uf.parentId[i])
    end
    return uf.parentId[i]
end

"""Retourne 'true' ssi i est le parent de j"""
function union(uf::UnionFind, i::Int, j::Int)
    iRoot = find(uf, i)
    jRoot = find(uf, j)
    if iRoot != jRoot
        if uf.rank[iRoot] < uf.rank[jRoot]
            uf.parentId[iRoot] = jRoot
            return false
        else
            uf.parentId[jRoot] = iRoot
            if uf.rank[iRoot] == uf.rank[jRoot]
                uf.rank[iRoot] += 1
            end
            return true
        end
    end
end