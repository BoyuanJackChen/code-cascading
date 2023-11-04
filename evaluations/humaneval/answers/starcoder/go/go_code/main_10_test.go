package main 
 
import ( "testing"
         "github.com/stretchr/testify/assert"
)

func TestMakePalindrome(t *testing.T) {
    assert := assert.New(t)
    assert.Equal("", MakePalindrome(""))
    assert.Equal("x", MakePalindrome("x"))
    assert.Equal("xyzyx", MakePalindrome("xyz"))
    assert.Equal("xyx", MakePalindrome("xyx"))
    assert.Equal("jerryrrej", MakePalindrome("jerry"))
}
