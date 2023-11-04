package main 
 
import ( "testing"
         "github.com/stretchr/testify/assert"
)

func TestSimplify(t *testing.T) {
    assert := assert.New(t)
    assert.Equal(true, Simplify("1/5", "5/1"))
    assert.Equal(false, Simplify("1/6", "2/1"))
    assert.Equal(true, Simplify("5/1", "3/1"))
    assert.Equal(false, Simplify("7/10", "10/2"))
    assert.Equal(true, Simplify("2/10", "50/10"))
    assert.Equal(true, Simplify("7/2", "4/2"))
    assert.Equal(true, Simplify("11/6", "6/1"))
    assert.Equal(false, Simplify("2/3", "5/2"))
    assert.Equal(false, Simplify("5/2", "3/5"))
    assert.Equal(true, Simplify("2/4", "8/4"))
    assert.Equal(true, Simplify("2/4", "4/2"))
    assert.Equal(true, Simplify("1/5", "5/1"))
    assert.Equal(false, Simplify("1/5", "1/5"))
}
