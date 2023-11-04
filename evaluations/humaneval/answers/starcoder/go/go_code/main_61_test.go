package main 
 
import ( "testing"
         "github.com/stretchr/testify/assert"
)

func TestCorrectBracketing(t *testing.T) {
    assert := assert.New(t)
    assert.Equal(true, CorrectBracketing("()"))
    assert.Equal(true, CorrectBracketing("(()())"))
    assert.Equal(true, CorrectBracketing("()()(()())()"))
    assert.Equal(true, CorrectBracketing("()()((()()())())(()()(()))"))
    assert.Equal(false, CorrectBracketing("((()())))"))
    assert.Equal(false, CorrectBracketing(")(()"))
    assert.Equal(false, CorrectBracketing("("))
    assert.Equal(false, CorrectBracketing("(((("))
    assert.Equal(false, CorrectBracketing(")"))
    assert.Equal(false, CorrectBracketing("(()"))
    assert.Equal(false, CorrectBracketing("()()(()())())(()"))
    assert.Equal(false, CorrectBracketing("()()(()())()))()"))
}
