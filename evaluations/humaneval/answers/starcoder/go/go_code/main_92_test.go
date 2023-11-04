package main 
 
import ( "testing"
         "github.com/stretchr/testify/assert"
)

func TestAnyInt(t *testing.T) {
    assert := assert.New(t)
    assert.Equal(true, AnyInt(2, 3, 1))
    assert.Equal(false, AnyInt(2.5, 2, 3))
    assert.Equal(false, AnyInt(1.5, 5, 3.5))
    assert.Equal(false, AnyInt(2, 6, 2))
    assert.Equal(true, AnyInt(4, 2, 2))
    assert.Equal(false, AnyInt(2.2, 2.2, 2.2))
    assert.Equal(true, AnyInt(-4, 6, 2))
    assert.Equal(true, AnyInt(2, 1, 1))
    assert.Equal(true, AnyInt(3, 4, 7))
    assert.Equal(false, AnyInt(3.0, 4, 7))
}
