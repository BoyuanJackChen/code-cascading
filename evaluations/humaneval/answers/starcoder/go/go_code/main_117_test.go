package main 
 
import ( "testing"
         "github.com/stretchr/testify/assert"
)

func TestSelectWords(t *testing.T) {
    assert := assert.New(t)
    assert.Equal([]string{"little"}, SelectWords("Mary had a little lamb", 4))
    assert.Equal([]string{"Mary", "lamb"}, SelectWords("Mary had a little lamb", 3))
    assert.Equal([]string{}, SelectWords("simple white space", 2))
    assert.Equal([]string{"world"}, SelectWords("Hello world", 4))
    assert.Equal([]string{"Uncle"}, SelectWords("Uncle sam", 3))
    assert.Equal([]string{}, SelectWords("", 4))
    assert.Equal([]string{"b", "c", "d", "f"}, SelectWords("a b c d e f", 1))
}
