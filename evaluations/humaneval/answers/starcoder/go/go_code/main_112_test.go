package main 
 
import ( "testing"
         "github.com/stretchr/testify/assert"
)

func TestReverseDelete(t *testing.T) {
    assert := assert.New(t)
    assert.Equal([2]interface{}{"bcd", false}, ReverseDelete("abcde","ae"))
    assert.Equal([2]interface{}{"acdef", false}, ReverseDelete("abcdef", "b"))
    assert.Equal([2]interface{}{"cdedc", true}, ReverseDelete("abcdedcba","ab"))
    assert.Equal([2]interface{}{"dik", false}, ReverseDelete("dwik","w"))
    assert.Equal([2]interface{}{"", true}, ReverseDelete("a","a"))
    assert.Equal([2]interface{}{"abcdedcba", true}, ReverseDelete("abcdedcba",""))
    assert.Equal([2]interface{}{"abcdedcba", true}, ReverseDelete("abcdedcba","v"))
    assert.Equal([2]interface{}{"abba", true}, ReverseDelete("vabba","v"))
    assert.Equal([2]interface{}{"", true}, ReverseDelete("mamma","mia"))
}
