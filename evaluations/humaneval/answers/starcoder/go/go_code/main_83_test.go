package main 
 
import ( "testing"
         "github.com/stretchr/testify/assert"
)

func TestStartsOneEnds(t *testing.T) {
    assert := assert.New(t)
    assert.Equal(1, StartsOneEnds(1))
    assert.Equal(18, StartsOneEnds(2))
    assert.Equal(180, StartsOneEnds(3))
    assert.Equal(1800, StartsOneEnds(4))
    assert.Equal(18000, StartsOneEnds(5))
}
