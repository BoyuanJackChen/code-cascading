package main 
 
import ( "testing"
         "github.com/stretchr/testify/assert"
)

func TestSolve(t *testing.T) {
    assert := assert.New(t)
    assert.Equal("aSdF", Solve("AsDf"))
    assert.Equal("4321", Solve("1234"))
    assert.Equal("AB", Solve("ab"))
    assert.Equal("#A@c", Solve("#a@C"))
    assert.Equal("#aSDFw^45", Solve("#AsdfW^45"))
    assert.Equal("2@6#", Solve("#6@2"))
    assert.Equal("#$A^d", Solve("#$a^D"))
    assert.Equal("#CCC", Solve("#ccc"))
}
