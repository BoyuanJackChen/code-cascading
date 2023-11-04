package main 
 
import ( "testing"
         "github.com/stretchr/testify/assert"
)

func TestSameChars(t *testing.T) {
    assert := assert.New(t)
    assert.Equal(true, SameChars("eabcdzzzz", "dddzzzzzzzddeddabc"))
    assert.Equal(true, SameChars("abcd", "dddddddabc"))
    assert.Equal(true, SameChars("dddddddabc", "abcd"))
    assert.Equal(false, SameChars("eabcd", "dddddddabc"))
    assert.Equal(false, SameChars("abcd", "dddddddabcf"))
    assert.Equal(false, SameChars("eabcdzzzz", "dddzzzzzzzddddabc"))
    assert.Equal(false, SameChars("aabb", "aaccc"))
}
