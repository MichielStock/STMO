module Words

export getwords, hamming, getwordedges

using HTTP, STMO

"""Get all English words"""
function getwords()
    io = HTTP.get("https://github.com/dwyl/english-words/raw/master/words_alpha.txt")
    words = split(String(io.body), "\n") .|> rstrip .|> String
    return words
end

"""Get all English words of length `wordsize`"""
function getwords(wordsize::Integer)
    return filter(w -> length(w) == wordsize, getwords())
end

"""Returns all the weigthed edges between a list of `words` where the Hamming
distance is less than `cutoff`"""
function getwordedges(words, cutoff=5)
    edges = Tuple{Int,String,String}[]
    for (i, w1) in enumerate(words[1:end-1])
        for w2 in words[(i+1):end]
            d = hamming(w1, w2)
            d ≤ cutoff && push!(edges, (d, w1, w2))
        end
    end
    return edges
end

end #Words
